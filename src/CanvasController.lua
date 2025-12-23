--!native
local AssetService = game:GetService("AssetService")
local UserInputService = game:GetService("UserInputService")

local EngineModule = script:WaitForChild("NeuralNet")
local Engine = require(EngineModule).new()

local Canvas = script.Parent
local MainFrame = Canvas.Parent :: Frame
local ButtonFrame = MainFrame:WaitForChild("Button")
local InputBox = MainFrame:WaitForChild("Input") :: TextBox
local GuessLabel = MainFrame:WaitForChild("GuessLabel") :: TextLabel

local ClearBtn = ButtonFrame:WaitForChild("Clear") :: TextButton
local DebugBtn = ButtonFrame:WaitForChild("Debug") :: TextButton
local ScanBtn = ButtonFrame:WaitForChild("Scan") :: TextButton

local RES = 28
local CANVAS_RES = Vector2.new(RES, RES)
local editableImage = AssetService:CreateEditableImage({Size = CANVAS_RES})
Canvas.ImageContent = Content.fromObject(editableImage)

local isDrawing = false
local lastPos: Vector2? = nil

local function getGridPos(pos: Vector3)
	local ap, as = Canvas.AbsolutePosition, Canvas.AbsoluteSize
	if pos.X < ap.X or pos.X > ap.X+as.X or pos.Y < ap.Y or pos.Y > ap.Y+as.Y then return nil end
	return Vector2.new(math.floor((pos.X-ap.X)/as.X*RES), math.floor((pos.Y-ap.Y)/as.Y*RES))
end

UserInputService.InputBegan:Connect(function(i)
	if i.UserInputType == Enum.UserInputType.MouseButton1 then 
		isDrawing = true 
		lastPos = getGridPos(i.Position)
		if lastPos then 
			editableImage:DrawCircle(lastPos, 1, Color3.new(1,1,1), 0, Enum.ImageCombineType.Overwrite)
		end
	end
end)

UserInputService.InputEnded:Connect(function() isDrawing = false; lastPos = nil end)

UserInputService.InputChanged:Connect(function(i)
	if isDrawing and i.UserInputType == Enum.UserInputType.MouseMovement then
		local p = getGridPos(i.Position)
		if p and lastPos then 
			editableImage:DrawLine(lastPos, p, Color3.new(1,1,1), 0, Enum.ImageCombineType.Overwrite)
			editableImage:DrawCircle(p, 1, Color3.new(1,1,1), 0, Enum.ImageCombineType.Overwrite)
			lastPos = p 
		end
	end
end)

local function getProcessedPixels(): {number}?
	local buf = editableImage:ReadPixelsBuffer(Vector2.zero, CANVAS_RES)
	local raw = table.create(784)

	for i = 0, buffer.len(buf)-1, 4 do 
		table.insert(raw, buffer.readu8(buf, i)/255) 
	end

	local minX, minY, maxX, maxY = RES, RES, 0, 0
	local active = false

	for y = 0, RES-1 do
		for x = 0, RES-1 do
			if raw[y * RES + x + 1] > 0.1 then
				active = true
				if x < minX then minX = x end; if x > maxX then maxX = x end
				if y < minY then minY = y end; if y > maxY then maxY = y end
			end
		end
	end

	if not active then return nil end

	local w, h = maxX - minX + 1, maxY - minY + 1
	local scale = math.min(20/w, 20/h)

	local tempGrid = table.create(784, 0)

	local sumX, sumY, totalWeight = 0, 0, 0

	for y = 0, h - 1 do
		for x = 0, w - 1 do
			local oldIdx = (minY + y) * RES + (minX + x) + 1
			if raw[oldIdx] > 0.2 then
				local nx = x * scale
				local ny = y * scale

				for dy = 0, 1 do
					for dx = 0, 1 do
						local drawX, drawY = math.floor(nx + dx), math.floor(ny + dy)
						if drawX >= 0 and drawX < 28 and drawY >= 0 and drawY < 28 then
							local idx = drawY * RES + drawX + 1
							if tempGrid[idx] == 0 then
								tempGrid[idx] = 1.0
								sumX += drawX
								sumY += drawY
								totalWeight += 1
							end
						end
					end
				end
			end
		end
	end

	if totalWeight == 0 then return nil end

	local comX = sumX / totalWeight
	local comY = sumY / totalWeight

	local shiftX = 13.5 - comX
	local shiftY = 13.5 - comY

	local finalPixels = table.create(784, 0)

	for y = 0, RES-1 do
		for x = 0, RES-1 do
			local idx = y * RES + x + 1
			local val = tempGrid[idx]

			if val > 0 then
				local targetX = x + shiftX
				local targetY = y + shiftY

				local neighbors = {
					{0, 0, 0.6},
					{1, 0, 0.2}, {-1, 0, 0.2}, {0, 1, 0.2}, {0, -1, 0.2},
					{1, 1, 0.1}, {1, -1, 0.1}, {-1, 1, 0.1}, {-1, -1, 0.1}
				}

				for _, point in ipairs(neighbors) do
					local fx = math.floor(targetX + point[1])
					local fy = math.floor(targetY + point[2])

					if fx >= 0 and fx < RES and fy >= 0 and fy < RES then
						local fIdx = fy * RES + fx + 1
						finalPixels[fIdx] = math.min(finalPixels[fIdx] + point[3], 1.0)
					end
				end
			end
		end
	end

	return finalPixels
end

local function onScan()
	local pixels = getProcessedPixels()
	if not pixels then 
		return 
	end

	local digit, confidence = Engine:Identify(pixels)
	GuessLabel.Text = string.format("Detected: %d (%d%%)", digit, confidence)

	if confidence < 50 then
		GuessLabel.TextColor3 = Color3.new(1, 0.5, 0.5)
	else
		GuessLabel.TextColor3 = Color3.new(0.5, 1, 0.5)
	end
end

local function debugShowProcessed()
	local pixels = getProcessedPixels()
	if not pixels then return end

	local buf = buffer.create(RES * RES * 4)

	for i, intensity in ipairs(pixels) do
		local val = math.floor(intensity * 255)

		local offset = (i - 1) * 4
		buffer.writeu8(buf, offset, val)
		buffer.writeu8(buf, offset + 1, val)
		buffer.writeu8(buf, offset + 2, val)
		buffer.writeu8(buf, offset + 3, 255)
	end

	editableImage:WritePixelsBuffer(Vector2.zero, CANVAS_RES, buf)
end

DebugBtn.MouseButton1Click:Connect(debugShowProcessed)

ScanBtn.MouseButton1Click:Connect(onScan)

ClearBtn.MouseButton1Click:Connect(function()
	editableImage:DrawRectangle(Vector2.zero, CANVAS_RES, Color3.new(0,0,0), 0, Enum.ImageCombineType.Overwrite)
	GuessLabel.Text = "Canvas Cleared"
	InputBox.Text = ""
end)
