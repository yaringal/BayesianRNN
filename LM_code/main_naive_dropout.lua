-- Author: original with my comments, sequences init with 0 rather than prev state
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

-- Note: original code has lr/decay**(max_max_epoch-max_epoch-1) = 0.004 to fine-tune error

-- -- param 3:
-- local params = {batch_size=20,
--                 seq_length=20,
--                 layers=2,
--                 decay=2,
--                 rnn_size=200,
--                 dropout=0.5,
--                 init_weight=0.1,
--                 lr=1,
--                 vocab_size=10000,
--                 max_epoch=8, -- start decreasing learning rate
--                 max_max_epoch=30,
--                 max_grad_norm=5}

-- param 4:
-- local params = {batch_size=20,
--                 seq_length=20,
--                 layers=2,
--                 decay=2,
--                 rnn_size=200,
--                 dropout=0.5,
--                 init_weight=0.1,
--                 lr=1,
--                 vocab_size=10000,
--                 max_epoch=21, -- start decreasing learning rate, keeping lr > 0.004
--                 max_max_epoch=30,
--                 max_grad_norm=5}

-- param 5:
-- local params = {batch_size=20,
--                 seq_length=20,
--                 layers=2,
--                 decay=1.3, -- 1./1.3**(30-8-1) = 0.004
--                 rnn_size=200,
--                 dropout=0.5,
--                 init_weight=0.1,
--                 lr=1,
--                 vocab_size=10000,
--                 max_epoch=8, -- start decreasing learning rate, keeping lr > 0.004
--                 max_max_epoch=30,
--                 max_grad_norm=5}

-- param 6:
-- local params = {batch_size=20,
--                 seq_length=20,
--                 layers=2,
--                 decay=1.05, -- 1./1.05**(180-65-1)
--                 rnn_size=200,
--                 dropout=0.5,
--                 init_weight=0.1,
--                 lr=1,
--                 vocab_size=10000,
--                 max_epoch=65, -- start decreasing learning rate, keeping lr > 0.004
--                 max_max_epoch=180,
--                 max_grad_norm=5}

-- param 7:
-- local params = {batch_size=20,
--                 seq_length=20,
--                 layers=2,
--                 decay=1.,
--                 rnn_size=200,
--                 dropout=0.5,
--                 init_weight=0.1,
--                 lr=1,
--                 vocab_size=10000,
--                 max_epoch=180, -- start decreasing learning rate, keeping lr > 0.004
--                 max_max_epoch=180,
--                 max_grad_norm=5}

-- -- param 8:
-- local params = {batch_size=20,
--                 seq_length=20,
--                 layers=2,
--                 decay=1.,
--                 rnn_size=200,
--                 dropout=0,
--                 init_weight=0.1,
--                 lr=1,
--                 vocab_size=10000,
--                 max_epoch=180, -- start decreasing learning rate, keeping lr > 0.004
--                 max_max_epoch=180,
--                 max_grad_norm=5}

-- -- param 9:
-- local params = {batch_size=20,
--                 seq_length=20,
--                 layers=2,
--                 decay=1.,
--                 rnn_size=200,
--                 dropout=0.5,
--                 init_weight=0.1,
--                 lr=0.1,
--                 vocab_size=10000,
--                 max_epoch=180, -- start decreasing learning rate
--                 max_max_epoch=180,
--                 max_grad_norm=5}

-- -- param 10:
-- local params = {batch_size=20,
--                 seq_length=20,
--                 layers=2,
--                 decay=1.,
--                 rnn_size=200,
--                 dropout=0.5,
--                 init_weight=0.1,
--                 lr=0.5,
--                 vocab_size=10000,
--                 max_epoch=180, -- start decreasing learning rate
--                 max_max_epoch=180,
--                 max_grad_norm=5}

-- -- param 11:
-- local params = {batch_size=20,
--                 seq_length=20,
--                 layers=2,
--                 decay=1.,
--                 rnn_size=200,
--                 dropout=0,
--                 init_weight=0.1,
--                 lr=0.5,
--                 vocab_size=10000,
--                 max_epoch=180, -- start decreasing learning rate
--                 max_max_epoch=180,
--                 max_grad_norm=5}

-- param 12:
local params = {batch_size=20,
                seq_length=20,
                layers=2,
                decay=1.,
                rnn_size=200,
                dropout=0.25,
                init_weight=0.1,
                lr=1,
                vocab_size=10000,
                max_epoch=180, -- start decreasing learning rate, keeping lr > 0.004
                max_max_epoch=180,
                max_grad_norm=5}

local function transfer_data(x)
  return x:cuda()
end

local state_train, state_valid, state_test
local model = {}
local paramx, paramdx

local function lstm(x, prev_c, prev_h)
  -- Calculate all four gates in one go
  local i2h = nn.Linear(params.rnn_size, 4*params.rnn_size)(x)
  local h2h = nn.Linear(params.rnn_size, 4*params.rnn_size)(prev_h)
  local gates = nn.CAddTable()({i2h, h2h})
  
  -- Reshape to (batch_size, n_gates, hid_size)
  -- Then slice the n_gates dimension, i.e dimension 2
  local reshaped_gates =  nn.Reshape(4,params.rnn_size)(gates)
  local sliced_gates = nn.SplitTable(2)(reshaped_gates)
  
  -- Use select gate to fetch each gate and apply nonlinearity
  local in_gate          = nn.Sigmoid()(nn.SelectTable(1)(sliced_gates))
  local in_transform     = nn.Tanh()(nn.SelectTable(2)(sliced_gates))
  local forget_gate      = nn.Sigmoid()(nn.SelectTable(3)(sliced_gates))
  local out_gate         = nn.Sigmoid()(nn.SelectTable(4)(sliced_gates))

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
  local i                = {[0] = LookupTable(params.vocab_size,
                                              params.rnn_size)(x)}
  local next_s           = {}
  local split            = {prev_s:split(2 * params.layers)}
  for layer_idx = 1, params.layers do
    local prev_c         = split[2 * layer_idx - 1]
    local prev_h         = split[2 * layer_idx]
    local dropped        = nn.Dropout(params.dropout)(i[layer_idx - 1])
    local next_c, next_h = lstm(dropped, prev_c, prev_h)
    table.insert(next_s, next_c)
    table.insert(next_s, next_h)
    i[layer_idx] = next_h
  end
  local h2y              = nn.Linear(params.rnn_size, params.vocab_size)
  local dropped          = nn.Dropout(params.dropout)(i[params.layers])
  local pred             = nn.LogSoftMax()(h2y(dropped))
  -- Author: By default, the losses are averaged over observations for each minibatch
  local err              = nn.ClassNLLCriterion()({pred, y})
  local module           = nn.gModule({x, y, prev_s},
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
  model.core_network = core_network
  model.rnns = g_cloneManyTimes(core_network, params.seq_length) -- Author: Are the parameters tied?
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

local function fp(state)
  g_replace_table(model.s[0], model.start_s) -- Author: wouldn't it make more sense to call this after the reset?
  if state.pos + params.seq_length > state.data:size(1) then
    reset_state(state)
  end
  for i = 1, params.seq_length do
    local x = state.data[state.pos]
    local y = state.data[state.pos + 1]
    local s = model.s[i - 1]
    model.err[i], model.s[i] = unpack(model.rnns[i]:forward({x, y, s}))
    state.pos = state.pos + 1
  end
  -- Author: we do not keep the last state to init the next sequence, but keep it at zero
  -- g_replace_table(model.start_s, model.s[params.seq_length])
  return model.err:mean()
end

local function bp(state)
  -- Author: we truncate the derivative at seq_length, which is equivalent
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
    local tmp = model.rnns[i]:backward({x, y, s},
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
end

local function run_valid()
  reset_state(state_valid)
  g_disable_dropout(model.rnns)
  local len = (state_valid.data:size(1) - 1) / (params.seq_length)
  local perp = 0
  for i = 1, len do
    perp = perp + fp(state_valid)
  end
  print("Validation set perplexity : " .. g_f3(torch.exp(perp / len)))
  g_enable_dropout(model.rnns)
end

local function run_test()
  reset_state(state_test)
  g_disable_dropout(model.rnns)
  local perp = 0
  local len = state_test.data:size(1)
  g_replace_table(model.s[0], model.start_s)
  for i = 1, (len - 1) do
    local x = state_test.data[i]
    local y = state_test.data[i + 1]
    perp_tmp, model.s[1] = unpack(model.rnns[1]:forward({x, y, model.s[0]}))
    perp = perp + perp_tmp[1]
    g_replace_table(model.s[0], model.s[1])
  end
  print("Test set perplexity : " .. g_f3(torch.exp(perp / (len - 1))))
  g_enable_dropout(model.rnns)
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
    local perp = fp(state_train) -- Author: perp is average of neg log probs over mini batch
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
  print("Training is over.")
end

main()
