------------------------------------------------------------------------
--[[ VRClassReward ]]--
-- Variance reduced classification reinforcement criterion.
-- input : {class prediction, baseline reward}
-- Reward is 1 for success, Reward is 0 otherwise.
-- reward = scale*(Reward - baseline) where baseline is 2nd input element
-- Note : for RNNs with R = 1 for last step in sequence, encapsulate it
-- in nn.ModuleCriterion(VRClassReward, nn.SelectTable(-1))
------------------------------------------------------------------------
local ChangeReward, parent = torch.class("nn.ChangeReward", "nn.Criterion")
local dbg = require("debugger")

function ChangeReward:__init(module, scale, criterion)
   parent.__init(self)
   self.module = module -- so it can call module:reinforce(reward)
   self.scale = scale or 1 -- scale of reward
   self.criterion = criterion or nn.MSECriterion() -- baseline criterion
   self.sizeAverage = true
   self.gradInput = {torch.Tensor()}
   self.std = 0
end

function ChangeReward:updateOutput(currentClassifierOutput, target)
--   assert(torch.type(input) == 'table')
--   local input = self:toBatch(input[1], 1)
--   self._maxVal = self._maxVal or input.new()
--   self._maxIdx = self._maxIdx or torch.type(input) == 'torch.CudaTensor' and input.new() or torch.LongTensor()
--
--   -- max class value is class prediction
--   self._maxVal:max(self._maxIdx, input, 2)
--   if torch.type(self._maxIdx) ~= torch.type(target) then
--      self._target = self._target or self._maxIdx.new()
--      self._target:resize(target:size()):copy(target)
--      target = self._target
--   end
--
--   -- reward = scale when correctly classified
--   self._reward = self._maxIdx.new()
--   self._reward:eq(self._maxIdx, target)
--   self.reward = self.reward or input.new()
--   self.reward:resize(self._reward:size(1)):copy(self._reward)
--   self.reward:mul(self.scale)
--
--   -- loss = -sum(reward)
--   self.output = -self.reward:sum()
--   if self.sizeAverage then
--      self.output = self.output/input:size(1)
--   end
--   return self.

   -- Norm implementation

   self.classifierOutput = self.classifierOutput or currentClassifierOutput:clone()  -- set the classifier output to itself or, if not defined already, to the input so the 
                                                                   -- norm will be zero for the first epoch
   local diff = currentClassifierOutput - self.classifierOutput -- calculate the difference between the current and previous classifier outputs
   self.reward = torch.norm(diff, 2, 2):squeeze() * .1-- calculate the norm of the difference of the two outputs and scale it to be used as the new reward
   local norm = self.reward:clone()
   difference = norm - (self.norm or norm)
   -- print (torch.mean(difference))
   self.norm = norm
   
   -- Standard deviation implementation 

   -- local std = torch.std(input)
   -- print (std - self.std)
   -- self.reward = torch.Tensor(20)
   -- self.reward:fill(std)
   -- self.reward = self.reward * .001
   -- self.std = torch.abs(std)

end

function ChangeReward:updateGradInput(input, target)
--

--   local input = self:toBatch(inputTable[1], 1)
--   local baseline = self:toBatch(inputTable[2], 1)
--
--   -- reduce variance of reward using baseline
--   self.vrReward = self.vrReward or self.reward.new()
--   self.vrReward:resizeAs(self.reward):copy(self.reward)
--   self.vrReward:add(-1, baseline)
--   if self.sizeAverage then
--      self.vrReward:div(input:size(1))
--   end

   self.module:reinforce(self.reward) -- broadcast reward to modules
--
--   -- zero gradInput (this criterion has no gradInput for class pred)
--   self.gradInput[1]:resizeAs(input):zero()
--   self.gradInput[1] = self:fromBatch(self.gradInput[1], 1)
--
--   -- learn the baseline reward
--   self.gradInput[2] = self.criterion:backward(baseline, self.reward)
--   self.gradInput[2] = self:fromBatch(self.gradInput[2], 1)
--   return self.gradInput
end

function ChangeReward:type(type)
   self._maxVal = nil
   self._maxIdx = nil
   self._target = nil
   local module = self.module
   self.module = nil
   local ret = parent.type(self, type)
   self.module = module
   return ret
end
