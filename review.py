import torch
import torchvision

cubic=torch.arange(0,27,1).view(3,3,3)
print(cubic)
print(torch.min(cubic,dim=0)[0])
print(torch.min(cubic,dim=1)[0])
print(torch.min(cubic,dim=2)[0])



a=torch.Tensor([0,1,2,3,4,5,6,7,8,9])
b=torch.Tensor([0,1,2,3,4,5,6,7,8,9])

b[0]+=1

b[4]+=1

c=torch.arange(100,1000+1,100)


z=torch.zeros(10)
z[a==b]=1

print(z)

v=torch.zeros(10)
v[a%2==0]=1

print(v)
print(v[::2])


v=torch.zeros(10)
v[a%3==0]=1

print(v)
print(v[::3])



N=10
C=64
HW=512

HWout=32
cnn_feature_map = torch.rand(N,C,HW,HW)*10




sel_grid=torch.zeros(HWout,HWout,2)
for y in range(HWout):
	for x in range(HWout):
		sel_grid[y][x][0]=(x-HW/2)/(HW/2)
		sel_grid[y][x][1]=(y-HW/2)/(HW/2)
print(sel_grid.shape)
print(sel_grid[HWout//2][HWout//2])

print(sel_grid[None,...].shape)


crop_feature_map=torch.zeros(N,C,HWout,HWout)
for i in range(cnn_feature_map.size(0)):
	crop_feature_map[i] = torch.nn.functional.grid_sample(cnn_feature_map[i:i+1], sel_grid[None,...],mode='nearest',align_corners=True)[0]



B_index=7
C_index=62
XY=21


temp=crop_feature_map[B_index][C_index][0][0]


for x in range(HW):
	for y in range(HW):
		if(cnn_feature_map[B_index][C_index][y][x]==temp):
			print(x,y)







print(crop_feature_map[B_index][C_index])
print(cnn_feature_map[B_index][C_index])
print(cnn_feature_map[B_index][C_index][XY][XY])
print(crop_feature_map[B_index][C_index][XY][XY])



print('roi--'*20)

cnn_feature_map = torch.rand(1,1,4,4)*10

boxes=torch.zeros(1,4)
for i in range(boxes.size(0)):
	boxes[i][0]=0
	boxes[i][1]=0
	boxes[i][2]=3
	boxes[i][3]=3

print(boxes)

feature_map_roi=torchvision.ops.roi_pool(cnn_feature_map,[boxes],(4,4))


temp=feature_map_roi[0][0][0][0]


print(cnn_feature_map.mean())
print(cnn_feature_map)
print(feature_map_roi)

for y in range(3):
	for x in range(3):
		if(cnn_feature_map[B_index][C_index][y][x]==temp):
			print(x,y)


print(feature_map_roi.shape)







# B_index=0
# C_index=2

# boxes = torch.rand(10, 4) * 100
# # they need to be in [x0, y0, x1, y1] format
# boxes[:, 2:] += boxes[:, :2]
# # create a random image
# image = torch.rand(1, 3, 200, 200)
# # extract regions in `image` defined in `boxes`, rescaling
# # them to have a size of 3x3
# pooled_regions = torchvision.ops.roi_align(image, [boxes], output_size=(3, 3))
	

# print(boxes.shape)
# # check the size
# print(pooled_regions[B_index][C_index])
# # torch.Size([10, 3, 3, 3])
# # or compute the intersection over union between
# # all pairs of boxes
# print(torchvision.ops.box_iou(boxes, boxes).shape)
# # torch.Size([10, 10])