import matplotlib.pyplot as plt
import numpy as np

img = plt.imread('./numpy-learn/003.jpg')
#plt.imshow(img)
#plt.show()

print(img.shape)
#(7360, 4912, 3) 图片的宽、高、颜色通道

#图片行方向合并
big_img = np.concatenate([img, img], axis=1)
# plt.imshow(big_img)
# plt.show()

#图片压缩,每10个像素压缩成一个像素
small_img = img[::10, ::10, :]
# plt.imshow(small_img)
# plt.show()

#逆序转换，图片上下颠倒
# plt.imshow(img[::-1, :, :])
# plt.show()
#逆序转换，图片颜色颠倒
# plt.imshow(img[:, :, ::-1])
# plt.show()

s_img = np.split(img, 2)
plt.imshow(s_img[0])
plt.show()