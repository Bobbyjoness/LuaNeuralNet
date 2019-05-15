-- Author: David Serrano (serranod7 | BobbyJones)
-- inspired by https://gist.github.com/cassiozen/de0dff87eb7ed599b5d0

local NeuralNetwork = {}

local function sigmoid(x)
	return 1 / (1 + math.exp(-x))
end

function NeuralNetwork.create(inputs, outputs, hiddenLayers, neuronsPerHiddenLayer, learningRate, initialWeightRange )
	local network = {}
	local inputs = inputs or 1
	local outputs = outputs or 1
	local hiddenLayers = hiddenLayers or math.ceil(inputs/2)
	local neuronsPerHiddenLayer = neuronsPerHiddenLayer or math.ceil((inputs*2/3) + outputs)
	local initialWeightRange = initialWeightRange or 1
	network.learningRate = learningRate

	network = setmetatable(network, {
		__index = NeuralNetwork
	})

	network[1] = {}
	for i = 1, inputs do
		network[1][i] = {}
	end

	for i = 2, hiddenLayers + 2 do
		network[i] = {}
		local neurons = neuronsPerHiddenLayer
		if i == hiddenLayers+2 then
			neurons = outputs
		end
		for j = 1, neurons do
			network[i][j] = {bias = math.random()*2*initialWeightRange - initialWeightRange}
			local numWeightedInputs = #network[i-1] 
			for k = 1, numWeightedInputs do
				network[i][j][k] = math.random()*2-1
			end
		end

	end
	return network
end

function NeuralNetwork:forwardPropagate(...)
	if (type(select(1, ...)) ~= "table" and select('#',...) ~= #self[1]) then
		error("Neural Network received "..select('#',...).." input[s] (expected "..#self[1].." input[s])",2)
	end
	if (type(select(1,...)) == "table" and #select(1, ...) ~= #self[1]) then
		error("Neural Network received "..#select(1,...).." input[s] (expected "..#self[1].." input[s])",2)
	end

	local inputs

	if type(select(1, ...)) == "table" then
		inputs = select(1, ...)
	else
		inputs = {select(1,...)}
	end

	local outputs = {}
	for i = 1, #self do
		for j = 1, #self[i] do
			if i == 1 then
				self[i][j].result = inputs[j]
			else
				self[i][j].result = self[i][j].bias 
				for k = 1, #self[i][j] do
					self[i][j].result = self[i][j].result + (self[i][j][k]*self[i-1][k].result)
				end
				self[i][j].result = sigmoid(self[i][j].result)
				if i == #self then
					table.insert(outputs, self[i][j].result)
				end
			end
		end
	end
	return self:softmax(outputs)
end

function NeuralNetwork:backwardPropagate(inputs,  desiredOutputs)
	assert(#inputs == #self[1], "Neural Network received ".. #inputs.." input[s] (expected "..#self[1].." input[s])")
	assert(#desiredOutputs == #self[#self], "Neural Network received "..#desiredOutputs.." desired output[s] (expected "..#self[#self].." desired output[s])")

	self:forwardPropagate(inputs)
	for i = #self, 2, -1 do
		local temp = {}
		for j = 1, #self[i] do
			if i == #self then
				self[i][j].delta = (desiredOutputs[j] - self[i][j].result) * self[i][j].result * (1 - self[i][j].result)
			else
				local weightDelta = 0
				for k = 1, #self[i+1] do
					weightDelta = weightDelta + self[i+1][k][j] * self[i+1][k].delta
				end
				self[i][j].delta = self[i][j].result * (1 - self[i][j].result) * weightDelta
			end
		end
	end
	for i = 2, #self do
		for j = 1, #self[i] do
			self[i][j].bias = self[i][j].delta * self.learningRate
			for k = 1, #self[i][j] do 
				self[i][j][k] = self[i][j][k] + self[i][j].delta * self.learningRate * self[i-1][k].result
			end
		end
	end
end

function NeuralNetwork:softmax(inputs)
	local sum = 0
	for i, val in ipairs (inputs) do
		sum = sum + math.exp(inputs[i])
	end

	local out = {}
	for i, val in ipairs (inputs) do
		out[i] = math.exp(inputs[i])/sum
	end

	return out
end

return NeuralNetwork
