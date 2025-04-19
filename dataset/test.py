from tartanair import TartanAirDataset

dataset = TartanAirDataset(root_dir="data/image_left")

print(len(dataset))


image1, image2, translation_vector, rotation_quaternion = dataset[0]

print(image1.shape, image2.shape, translation_vector, rotation_quaternion)
