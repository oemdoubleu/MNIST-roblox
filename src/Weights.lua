local HttpService = game:GetService("HttpService")
local jsonString = [[ 
-- put your weights here
]]
if #jsonString < 10 then
	return nil
end
return HttpService:JSONDecode(jsonString)
