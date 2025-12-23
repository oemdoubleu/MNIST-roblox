--!native
--!optimize 2
local NeuralNet = {}
NeuralNet.__index = NeuralNet

local PreTrainedData = require(script.Parent:WaitForChild("Weights"))

local INPUT_SIZE = 784
local HIDDEN_SIZE = 128
local OUTPUT_SIZE = 10

function NeuralNet.new()
	local self = setmetatable({}, NeuralNet)

	if not PreTrainedData or not PreTrainedData.W1 then
		error("MISSING WEIGHTS")
	end

	self.W1 = PreTrainedData.W1
	self.B1 = PreTrainedData.B1
	self.W2 = PreTrainedData.W2
	self.B2 = PreTrainedData.B2

	return self
end

local function relu(x: number): number
	return if x > 0 then x else 0
end

function NeuralNet:Identify(inputs: {number})
	local hiddenNeurons = table.create(HIDDEN_SIZE, 0)

	for j = 1, HIDDEN_SIZE do
		local sum = self.B1[j]
		for i = 1, INPUT_SIZE do
			if inputs[i] > 0 then
				local wIndex = (i - 1) * HIDDEN_SIZE + j
				sum += inputs[i] * self.W1[wIndex]
			end
		end
		hiddenNeurons[j] = relu(sum)
	end

	local finalOutputs = table.create(OUTPUT_SIZE, 0)

	for k = 1, OUTPUT_SIZE do
		local sum = self.B2[k]
		for j = 1, HIDDEN_SIZE do
			if hiddenNeurons[j] > 0 then
				local wIndex = (j - 1) * OUTPUT_SIZE + k
				sum += hiddenNeurons[j] * self.W2[wIndex]
			end
		end
		finalOutputs[k] = sum
	end

	local max = -math.huge
	for _, v in finalOutputs do if v > max then max = v end end

	local expSum = 0
	local probs = table.create(OUTPUT_SIZE)
	for i, v in finalOutputs do
		local e = math.exp(v - max)
		probs[i] = e
		expSum += e
	end

	local bestDigit = 0
	local maxProb = -1
	for i = 1, OUTPUT_SIZE do
		local p = probs[i] / expSum
		if p > maxProb then
			maxProb = p
			bestDigit = i - 1
		end
	end

	return bestDigit, math.floor(maxProb * 100)
end

return NeuralNet
