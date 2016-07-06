-- This script implements Variational LSTM (Gal, 2015) for the large model of Zaremba et al. (2014).
-- In the setting of Zaremba et al. the states are not reset and the testing is done with a single 
-- pass through the test set. The only changes I've made to the setting of Zaremba et al. are:
-- 1. dropout technique (using a Bayesian LSTM)
-- 2. weight decay (which was chosen to be zero in Zaremba et al.)
-- All other hypers being identical to Zaremba et al.: learning rate decay was not tuned for my setting 
-- and is used following Zaremba et al., and the sequences are initialised with the previous state following 
-- Zaremba et al. (unlike in main_dropout.lua). Dropout parameters were optimised with grid search 
-- (tying dropout_x & dropout_h and dropout_i & dropout_o) over validation perplexity (optimal values 
-- are 0.3 and 0.5 compared Zaremba et al.'s 0.6). Note that unlike main_new_dropout_SOTA.lua, this script
-- does not use a reduced model size, but the same size model as in Zaremba et al..
-- 
-- Single model test perplexity is improved from Zaremba et al.'s 78.4 to 73.4 (using MC dropout at test time) 
-- and 75.2 with the dropout approximation. Validation perplexity is reduced from 82.2 to 77.9.
-- 
-- This script implements MC sampling at test time (printing predicted *log* probabilities obtained from 
-- stochastic forward passes for the entire test set many (MC_samples to be exact) times). Note that the 
-- resulting log file can get very large and that each test point is loaded into a batch and repeated 20 times
-- (meaning that the log would be of the form log p_1,1, log p_1,2, ..., log p_1,20, log p_2,1, ..., log p_2,20, 
-- ..., log p_1,21, ...).
-- To obtain the results in the paper you would need to exponentiate the log probs, and average the probabilities 
-- themselves (rather than the log probabilities!) for each point p_i. Note that the results in the paper were 
-- obtained by repeating this experiment 3 times and averaging the resulting perplexity (discarding numerically 
-- in-stable runs).
-- 
-- References:
-- Gal, Y, "A Theoretically Grounded Application of Dropout in Recurrent Neural Networks", 2015.
-- Zaremba, W, Sutskever, I, Vinyals, O, "Recurrent neural network regularization", 2014.
local ok,cunn = pcall(require, 'fbcunn')
if not ok then
    ok,cunn = pcall(require,'cunn')
    if ok then
        print("warning: fbcunn not found. Falling back to cunn") 
        LookupTable = nn.LookupTable
    else
        print("Could not find cunn or fbcunn. Either is required")
        os.exit()
    end
else
    deviceParams = cutorch.getDeviceProperties(1)
    cudaComputeCapability = deviceParams.major + deviceParams.minor/10
    LookupTable = nn.LookupTable
end
require('nngraph')
require('base')
local ptb = require('data')

-- SOTA params: (MC testing, 1500 units)
local params = {batch_size=20,
                seq_length=35,
                layers=2,
                decay=1.15,
                rnn_size=1500,
                dropout_x=0.3,
                dropout_i=0.5,
                dropout_h=0.3,
                dropout_o=0.5,
                init_weight=0.04,
                lr=1,
                vocab_size=10000,
                max_epoch=10,
                max_max_epoch=55,
                max_grad_norm=10,
                weight_decay=1e-7,
                MC_samples=100 -- number of outputs for each point is MC_samples x batch_size
              }

-- Yarin: use dropout from within the script rather than nn's
local disable_dropout = false
local function local_Dropout(input, noise)
  return nn.CMulTable()({input, noise})
end

local function transfer_data(x)
  return x:cuda()
end

local state_train, state_valid, state_test
local model = {}
local paramx, paramdx

local function lstm(x, prev_c, prev_h, noise_i, noise_h)
  -- Reshape to (batch_size, n_gates, hid_size)
  -- Then slice the n_gates dimension, i.e dimension 2
  local reshaped_noise_i = nn.Reshape(4,params.rnn_size)(noise_i)
  local reshaped_noise_h = nn.Reshape(4,params.rnn_size)(noise_h)
  local sliced_noise_i   = nn.SplitTable(2)(reshaped_noise_i)
  local sliced_noise_h   = nn.SplitTable(2)(reshaped_noise_h)
  -- Calculate all four gates 
  local i2h, h2h         = {}, {}
  for i = 1, 4 do 
    -- Use select table to fetch each gate
    local dropped_x      = local_Dropout(x, nn.SelectTable(i)(sliced_noise_i))
    local dropped_h      = local_Dropout(prev_h, nn.SelectTable(i)(sliced_noise_h))
    i2h[i]               = nn.Linear(params.rnn_size, params.rnn_size)(dropped_x)
    h2h[i]               = nn.Linear(params.rnn_size, params.rnn_size)(dropped_h)
  end
  
  -- Apply nonlinearity
  local in_gate          = nn.Sigmoid()(nn.CAddTable()({i2h[1], h2h[1]}))
  local in_transform     = nn.Tanh()(nn.CAddTable()({i2h[2], h2h[2]}))
  local forget_gate      = nn.Sigmoid()(nn.CAddTable()({i2h[3], h2h[3]}))
  local out_gate         = nn.Sigmoid()(nn.CAddTable()({i2h[4], h2h[4]}))

  local next_c           = nn.CAddTable()({
      nn.CMulTable()({forget_gate, prev_c}),
      nn.CMulTable()({in_gate,     in_transform})
  })
  local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

  return next_c, next_h
end

local function create_network()
  local x                = nn.Identity()()
  local y                = nn.Identity()()
  local prev_s           = nn.Identity()()
  local noise_x          = nn.Identity()()
  local noise_i          = nn.Identity()()
  local noise_h          = nn.Identity()()
  local noise_o          = nn.Identity()()
  local i                = {[0] = LookupTable(params.vocab_size,
                                              params.rnn_size)(x)}
  i[0] = local_Dropout(i[0], noise_x)
  local next_s           = {}
  local split            = {prev_s:split(2 * params.layers)}
  local noise_i_split    = {noise_i:split(params.layers)}
  local noise_h_split    = {noise_h:split(params.layers)}
  for layer_idx = 1, params.layers do
    local prev_c         = split[2 * layer_idx - 1]
    local prev_h         = split[2 * layer_idx]
    local n_i            = noise_i_split[layer_idx]
    local n_h            = noise_h_split[layer_idx]
    local next_c, next_h = lstm(i[layer_idx - 1], prev_c, prev_h, n_i, n_h)
    table.insert(next_s, next_c)
    table.insert(next_s, next_h)
    i[layer_idx] = next_h
  end
  local h2y              = nn.Linear(params.rnn_size, params.vocab_size)
  local dropped          = local_Dropout(i[params.layers], noise_o)
  local pred             = nn.LogSoftMax()(h2y(dropped))
  local err              = nn.ClassNLLCriterion()({pred, y})
  local module           = nn.gModule({x, y, prev_s, noise_x, noise_i, noise_h, noise_o},
                                      {err, nn.Identity()(next_s)})
  module:getParameters():uniform(-params.init_weight, params.init_weight)
  return transfer_data(module)
end

local function setup()
  print("Creating a RNN LSTM network.")
  local core_network = create_network()
  paramx, paramdx = core_network:getParameters()
  model.s = {}
  model.ds = {}
  model.start_s = {}
  for j = 0, params.seq_length do
    model.s[j] = {}
    for d = 1, 2 * params.layers do
      model.s[j][d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    end
  end
  for d = 1, 2 * params.layers do
    model.start_s[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    model.ds[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
  end
  -- YARIN: Note that the data comes in batches. We need noise to have batch by layers 
  -- by rnn_size dimensionality.
  model.noise_i = {}
  model.noise_x = {}
  model.noise_xe = {} -- Yarin: we expand the dims of noise_x to match data dim
  for j = 1, params.seq_length do
    model.noise_x[j] = transfer_data(torch.zeros(params.batch_size, 1))
    model.noise_xe[j] = torch.expand(model.noise_x[j], params.batch_size, params.rnn_size)
    model.noise_xe[j] = transfer_data(model.noise_xe[j])
  end
  model.noise_h = {}
  for d = 1, params.layers do
    model.noise_i[d] = transfer_data(torch.zeros(params.batch_size, 4 * params.rnn_size))
    model.noise_h[d] = transfer_data(torch.zeros(params.batch_size, 4 * params.rnn_size))
  end
  model.noise_o = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
  model.core_network = core_network
  model.rnns = g_cloneManyTimes(core_network, params.seq_length)
  model.norm_dw = 0
  model.err = transfer_data(torch.zeros(params.seq_length))
end

local function reset_state(state)
  state.pos = 1
  if model ~= nil and model.start_s ~= nil then
    for d = 1, 2 * params.layers do
      model.start_s[d]:zero()
    end
  end
end

local function reset_ds()
  for d = 1, #model.ds do
    model.ds[d]:zero()
  end
end

-- Yarin: convenience functions to handle noise
local function sample_noise(state)
  -- Yarin: assuming state.pos is at start of input sequence
  for i = 1, params.seq_length do
    -- Yarin: cheating here - sampling iid Berns for each x; should tie over words
    model.noise_x[i]:bernoulli(1 - params.dropout_x)
    model.noise_x[i]:div(1 - params.dropout_x)
  end
  -- Yarin: tying over words - overriding Berns for words that were already sampled. 
  -- this is efficient for short sequences, but longer ones it might be better to sample 
  -- once for all words.
  for b = 1, params.batch_size do
    for i = 1, params.seq_length do
      local x = state.data[state.pos + i - 1]
      for j = i+1, params.seq_length do
        if state.data[state.pos + j - 1] == x then
          model.noise_x[j][b] = model.noise_x[i][b]
          -- we only need to override the first time; afterwards subsequent are copied:
          break
        end
      end
    end
  end
  for d = 1, params.layers do
    model.noise_i[d]:bernoulli(1 - params.dropout_i)
    model.noise_i[d]:div(1 - params.dropout_i)
    model.noise_h[d]:bernoulli(1 - params.dropout_h)
    model.noise_h[d]:div(1 - params.dropout_h)
  end
  model.noise_o:bernoulli(1 - params.dropout_o)
  model.noise_o:div(1 - params.dropout_o)
end

local function reset_noise()
  for j = 1, params.seq_length do
    model.noise_x[j]:zero():add(1)
  end
  for d = 1, params.layers do
    model.noise_i[d]:zero():add(1)
    model.noise_h[d]:zero():add(1)
  end
  model.noise_o:zero():add(1)
end

local function fp(state)
  g_replace_table(model.s[0], model.start_s)
  if state.pos + params.seq_length > state.data:size(1) then
    reset_state(state)
  end
  -- Yarin: should reset noise out of function 
  if disable_dropout then reset_noise() else sample_noise(state) end
  for i = 1, params.seq_length do
    local x = state.data[state.pos]
    local y = state.data[state.pos + 1]
    local s = model.s[i - 1]
    model.err[i], model.s[i] = unpack(model.rnns[i]:forward(
      {x, y, s, model.noise_xe[i], model.noise_i, model.noise_h, model.noise_o}))
    state.pos = state.pos + 1
  end
  g_replace_table(model.start_s, model.s[params.seq_length])
  return model.err
end

local function bp(state)
  -- Yarin: we truncate the derivative at seq_length, which is equivalent
  -- to using sequences of length seq_length but with smarter initialisation
  -- than putting zeros for the first state. This is easier than bucketing,
  -- but carries internal states over <eos> which is bad. Especially because
  -- that means we use shorter sequences for each sentence. Note that it seems
  -- bad to reset ds if we use the prev s?
  paramdx:zero()
  reset_ds()
  for i = params.seq_length, 1, -1 do
    state.pos = state.pos - 1
    local x = state.data[state.pos]
    local y = state.data[state.pos + 1]
    local s = model.s[i - 1]
    local derr = transfer_data(torch.ones(1))
    local tmp = model.rnns[i]:backward( -- Yarin: do we need model.noise_x[i+1]?
      {x, y, s, model.noise_xe[i], model.noise_i, model.noise_h, model.noise_o},
      {derr, model.ds})[3]
    g_replace_table(model.ds, tmp)
    cutorch.synchronize()
  end
  state.pos = state.pos + params.seq_length
  model.norm_dw = paramdx:norm()
  if model.norm_dw > params.max_grad_norm then
    local shrink_factor = params.max_grad_norm / model.norm_dw
    paramdx:mul(shrink_factor)
  end
  paramx:add(paramdx:mul(-params.lr))
  -- Yarin: add weight decay
  paramx:add(-params.weight_decay, paramx)
end

local function run_valid()
  reset_state(state_valid)
  -- Yarin: disable dropout for standard dropout
  disable_dropout = true
  local len = (state_valid.data:size(1) - 1) / (params.seq_length)
  local perp = 0
  for i = 1, len do
    local p = fp(state_valid)
    perp = perp + p:mean()
  end
  print("Validation set perplexity : " .. g_f3(torch.exp(perp / len)))
  disable_dropout = false
end

local function run_test()
  reset_state(state_test)
  reset_noise()
  local perp = 0
  local len = state_test.data:size(1)
  g_replace_table(model.s[0], model.start_s)
  for i = 1, (len - 1) do
    local x = state_test.data[i]
    local y = state_test.data[i + 1]
    perp_tmp, model.s[1] = unpack(model.rnns[1]:forward(
      {x, y, model.s[0], model.noise_xe[1], model.noise_i, model.noise_h, model.noise_o}))
    perp = perp + perp_tmp[1]
    g_replace_table(model.s[0], model.s[1])
  end
  print("Test set perplexity : " .. g_f3(torch.exp(perp / (len - 1))))
end

local function run_MC_test()
  for sample = 1, params.MC_samples do
    reset_state(state_test)
    sample_noise(state_test)
    model.noise_x[1]:zero():add(1)
    local perp = 0
    local len = state_test.data:size(1)
    g_replace_table(model.s[0], model.start_s)
    for i = 1, (len - 1) do
      local x = state_test.data[i]
      local y = state_test.data[i + 1]
      perp_tmp, model.s[1] = unpack(model.rnns[1]:forward(
        {x, y, model.s[0], model.noise_xe[1], model.noise_i, model.noise_h, model.noise_o}))
      local pred_prob = model.rnns[1].outnode.data.mapindex[1].input[1]
      -- all state_test batch sequences are the same (we use different dropout mask for each one though)
      for j = 1, params.batch_size do 
        print(pred_prob[j][y[j]])
      end
      perp = perp + perp_tmp[1]
      g_replace_table(model.s[0], model.s[1])
    end
    print("Test set perplexity : " .. g_f3(torch.exp(perp / (len - 1))))
  end
end

local function main()
  g_init_gpu(arg)
  state_train = {data=transfer_data(ptb.traindataset(params.batch_size))}
  state_valid =  {data=transfer_data(ptb.validdataset(params.batch_size))}
  state_test =  {data=transfer_data(ptb.testdataset(params.batch_size))}
  print("Network parameters:")
  print(params)
  local states = {state_train, state_valid, state_test}
  for _, state in pairs(states) do
    reset_state(state)
  end
  setup()
  local step = 0
  local epoch = 0
  local total_cases = 0
  local beginning_time = torch.tic()
  local start_time = torch.tic()
  print("Starting training.")
  local epoch_size = torch.floor(state_train.data:size(1) / params.seq_length)
  local perps
  while epoch < params.max_max_epoch do
    local perp = fp(state_train):mean()
    if perps == nil then
      perps = torch.zeros(epoch_size):add(perp)
    end
    perps[step % epoch_size + 1] = perp
    step = step + 1
    bp(state_train)
    total_cases = total_cases + params.seq_length * params.batch_size
    epoch = step / epoch_size
    if step % torch.round(epoch_size / 10) == 10 then
      local wps = torch.floor(total_cases / torch.toc(start_time))
      local since_beginning = g_d(torch.toc(beginning_time) / 60)
      print('epoch = ' .. g_f3(epoch) ..
            ', train perp. = ' .. g_f3(torch.exp(perps:mean())) ..
            ', wps = ' .. wps ..
            ', dw:norm() = ' .. g_f3(model.norm_dw) ..
            ', lr = ' ..  g_f3(params.lr) ..
            ', since beginning = ' .. since_beginning .. ' mins.')
    end
    if step % epoch_size == 0 then
      run_valid()
      if epoch > params.max_epoch then
          params.lr = params.lr / params.decay
      end
    end
    if step % 33 == 0 then
      cutorch.synchronize()
      collectgarbage()
    end
  end
  run_test()
  run_MC_test()
  print("Training is over.")
end

main()
