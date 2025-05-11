from tartanair import TartanAirDataset

dataset = TartanAirDataset(root_dir="data", seq_len=2)

print(len(dataset))


images, translation_vector, rotation_quaternion = dataset[0]

print(images.shape, translation_vector.shape, rotation_quaternion.shape)
