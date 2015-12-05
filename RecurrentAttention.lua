------------------------------------------------------------------------
--[[ RecurrentAttention ]]-- 
-- Ref. A. http://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf
-- B. http://incompleteideas.net/sutton/williams-92.pdf
-- module which takes an RNN as argument with other 
-- hyper-parameters such as the maximum number of steps, 
-- action (actions sampling module like ReinforceNormal) and 
------------------------------------------------------------------------
local RecurrentAttention, parent = torch.class("nn.RecurrentAttention", "nn.AbstractSequencer")
local dbg = require("debugger")

function RecurrentAttention:__init(rnn, action, nStep, hiddenSize)
   parent.__init(self)
   assert(torch.isTypeOf(action, 'nn.Module'))
   assert(torch.type(nStep) == 'number')
   assert(torch.type(hiddenSize) == 'table')
   assert(torch.type(hiddenSize[1]) == 'number', "Does not support table hidden layers" )

   self.rnn = rnn
   -- we can decorate the module with a Recursor to make it AbstractRecurrent
   self.rnn = (not torch.isTypeOf(rnn, 'nn.AbstractRecurrent')) and nn.Recursor(rnn) or rnn
   -- backprop through time (BPTT) will be done online (in reverse order of forward)
   self.rnn:backwardOnline()
   for i,modula in ipairs(self.rnn:listModules()) do
      if torch.isTypeOf(modula, "nn.AbstractRecurrent") then
         modula.copyInputs = false
         modula.copyGradOutputs = false
      end
   end
   
   -- samples an x,y actions for each example
   self.action =  (not torch.isTypeOf(action, 'nn.AbstractRecurrent')) and nn.Recursor(action) or action 
   self.action:backwardOnline()
   self.hiddenSize = hiddenSize
   self.nStep = nStep
   
   self.modules = {self.rnn, self.action}
   
   self.output = {} -- rnn output
   self.actions = {} -- action output
   
   self.forwardActions = false
   
   self.gradHidden = {}
end

function RecurrentAttention:updateOutput(input)
   self.rnn:forget()
   self.action:forget()
   local nDim = input:dim()

   for step=1,self.nStep do
      if step == 1 then
         -- sample an initial starting actions by forwarding zeros through the action
         self._initInput = self._initInput or input.new()
         self._initInput:resize(input:size(1),table.unpack(self.hiddenSize)):zero()
         self.output[0] = self._initInput
--         self.actions[1] = self.action:updateOutput(self._initInput)
--      else
         -- sample actions from previous hidden activation (rnn output)
--         self.actions[step] = self.action:updateOutput(self.output[step-1])
      end
      self.actions[step] = self.action:updateOutput(self.output[step-1])

      -- rnn handles the recurrence internally
      local output = self.rnn:updateOutput{input, self.actions[step] }
      self.output[step] = self.forwardActions and {output, self.actions[step]} or output

      --[[ new code ]]--
      local classifierOutput = self.classifier:forward(self.output) -- get the current classification of the rnn
      --    ^^^^^^^^^^^^^^^^ this is NOT CHANGING for the first 5 glimpses
      self.rewardCriterion:updateOutput(classifierOutput) -- tell the criterion to calculate the reward for the locator
      self.rewardCriterion:updateGradInput(input, self.output) -- tell the criterion to broadcast its reward to the locator

      local currentModule = self.action:getStepModule(step) -- get the SEQUENTIAL module, not the Recursor
      currentModule:backward(self.output[step-1], torch.Tensor(output)) -- backpropagate and update weights
      --[[end new code]]


      --      self.action:backward(input, self.output[step]) --TODO: this input has to be the same input as the one originally fed to action (and I think it is)
--      self.inputs = self.inputs or {}
--      self.gradOutputs = self.gradOutputs or {}
--      self.inputs[step] = input
--      self.gradOutputs[step] = self.output
--      self.action:updateGradInputThroughTime(step+1, 1) --TODO: this input has to be the same input as the one originally fed to action (and I think it is)
--      self.action:updateGradInput(self.inputs, self.output)
      --TODO: also I don't know what self.output is doing here and this is probably a bad value. However, I think it can be a dummy value.
   end
   
   return self.output
end

function RecurrentAttention:updateGradInput(input, gradOutput)
   assert(self.rnn.step - 1 == self.nStep, "inconsistent rnn steps")
   assert(torch.type(gradOutput) == 'table', "expecting gradOutput table")
   assert(#gradOutput == self.nStep, "gradOutput should have nStep elements")
    
   -- back-propagate through time (BPTT)
   for step=self.nStep,1,-1 do
      -- 1. backward through the action layer
      local gradOutput_, gradAction_ = gradOutput[step]
      if self.forwardActions then
         gradOutput_, gradAction_ = unpack(gradOutput[step])
      else
         -- Note : gradOutput is ignored by REINFORCE modules so we give a zero Tensor instead
         self._gradAction = self._gradAction or self.action.output.new()
         if not self._gradAction:isSameSizeAs(self.action.output) then
            self._gradAction:resizeAs(self.action.output):zero()
         end
         gradAction_ = self._gradAction
      end
      
      if step == self.nStep then
         self.gradHidden[step] = nn.rnn.recursiveCopy(self.gradHidden[step], gradOutput_)
      else
         -- gradHidden = gradOutput + gradAction
         nn.rnn.recursiveAdd(self.gradHidden[step], gradOutput_)
      end

      if step == 1 then
         -- backward through initial starting actions
         self.action:updateGradInput(self._initInput, gradAction_)
      else
         local gradAction = self.action:updateGradInput(self.output[step-1], gradAction_)
         self.gradHidden[step-1] = nn.rnn.recursiveCopy(self.gradHidden[step-1], gradAction)
      end
      
      -- 2. backward through the rnn layer
      local gradInput = self.rnn:updateGradInput(input, self.gradHidden[step])[1]
      if step == self.nStep then
         self.gradInput:resizeAs(gradInput):copy(gradInput)
      else
         self.gradInput:add(gradInput)
      end
   end

   return self.gradInput
end

function RecurrentAttention:accGradParameters(input, gradOutput, scale)
   assert(self.rnn.step - 1 == self.nStep, "inconsistent rnn steps")
   assert(torch.type(gradOutput) == 'table', "expecting gradOutput table")
   assert(#gradOutput == self.nStep, "gradOutput should have nStep elements")
   
   -- back-propagate through time (BPTT)
   for step=self.nStep,1,-1 do
      -- 1. backward through the action layer
      local gradAction_ = self.forwardActions and gradOutput[step][2] or self._gradAction
            
      if step == 1 then
         -- backward through initial starting actions
         self.action:accGradParameters(self._initInput, gradAction_, scale)
      else
         self.action:accGradParameters(self.output[step-1], gradAction_, scale)
      end
      
      -- 2. backward through the rnn layer
      self.rnn:accGradParameters(input, self.gradHidden[step], scale)
   end
end

function RecurrentAttention:accUpdateGradParameters(input, gradOutput, lr)
   assert(self.rnn.step - 1 == self.nStep, "inconsistent rnn steps")
   assert(torch.type(gradOutput) == 'table', "expecting gradOutput table")
   assert(#gradOutput == self.nStep, "gradOutput should have nStep elements")
    
   -- backward through the action layers
   for step=self.nStep,1,-1 do
      -- 1. backward through the action layer
      local gradAction_ = self.forwardActions and gradOutput[step][2] or self._gradAction
      
      if step == 1 then
         -- backward through initial starting actions
         self.action:accUpdateGradParameters(self._initInput, gradAction_, lr)
      else
         -- Note : gradOutput is ignored by REINFORCE modules so we give action.output as a dummy variable
         self.action:accUpdateGradParameters(self.output[step-1], gradAction_, lr)
      end
      
      -- 2. backward through the rnn layer
      self.rnn:accUpdateGradParameters(input, self.gradHidden[step], lr)
   end
end

function RecurrentAttention:type(type)
   self._input = nil
   self._actions = nil
   self._crop = nil
   self._pad = nil
   self._byte = nil
   return parent.type(self, type)
end

function RecurrentAttention:__tostring__()
   local tab = '  '
   local line = '\n'
   local ext = '  |    '
   local extlast = '       '
   local last = '   ... -> '
   local str = torch.type(self)
   str = str .. ' {'
   str = str .. line .. tab .. 'action : ' .. tostring(self.action):gsub(line, line .. tab .. ext)
   str = str .. line .. tab .. 'rnn     : ' .. tostring(self.rnn):gsub(line, line .. tab .. ext)
   str = str .. line .. '}'
   return str
end
