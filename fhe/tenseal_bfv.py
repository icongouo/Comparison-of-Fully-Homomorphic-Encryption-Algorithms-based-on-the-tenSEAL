import tenseal as ts
from PIL import Image
import numpy as np
import time
import math
import os

public_context = ts.context(
    ts.SCHEME_TYPE.BFV,
    poly_modulus_degree=8192,
    plain_modulus=1032193
)

# 生成密钥
sk = public_context.secret_key()
public_context.generate_galois_keys()
public_context.generate_relin_keys()

time_list_encrypt = []
time_list_match = []
time_list_decrypt = []

# 分发所有密钥
key_folder = "./bfv_key"
key_path = f"{key_folder}/bfv_key.bin"
with open(key_path,'wb') as file:
    file.write(public_context.serialize(
        save_galois_keys=True,
        save_public_key=True,
        save_relin_keys=True,
        save_secret_key=True))

def image_read(image_path):
    # 由于算法的密文长度限制，用灰度形式读取图片
    img = Image.open(image_path).convert('L')
    # 将Image对象转换为一个Numpy数组，并展开成一维向量
    np_img = np.asarray(img).flatten()
    return np_img

# 定义函数将图片加密并保存
def encrypt_and_save_image(img_array, context, output_folder, index):
    # 加密图像
    start_time = time.time()
    enc_image = ts.bfv_vector(context, img_array)
    end_time = time.time()
    time1 = end_time - start_time
    time_list_encrypt.append(time1)
    print(f"encrypt NO.{index} image cost: {time1} seconds")

    # 保存加密数据到文件
    file_path = f"{output_folder}/data_{index}.bin"
    with open(file_path, 'wb') as file:
        file.write(enc_image.serialize())

    return file_path

# 设置输入输出文件夹路径
input_folder = "./input_images"  # 文件夹包含待加密的图片
output_folder = "./encrypted_images_bfv"  # 文件夹用于存储加密后的图片数据

# 创建输出文件夹（如果不存在）
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历文件夹下的所有图片并依次加密保存
print("-----------------encrypt images---------------------")
for index, filename in enumerate(os.listdir(input_folder)):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        image_path = os.path.join(input_folder, filename)
        img_array = image_read(image_path)
        encrypt_and_save_image(img_array, public_context, output_folder, index)
print(f"encrypt {len(time_list_encrypt)} images cost {sum(time_list_encrypt)} seconds")
print(f"encrypt a single image cost: {np.mean(time_list_encrypt)} seconds on average")

# 读取查询图片并加密
query_image_path = "4.png"  # 查询图片路径
query_image = Image.open(query_image_path).convert('L')
query_image_array = np.asarray(query_image)
query_image_array = query_image_array.flatten()
enc_query_image = ts.bfv_vector(public_context, query_image_array)

# 遍历存储的加密图片并计算相似度
print("-----------------matching images(close to 0 means similar)---------------------")
max_similarity = 100
max_similarity_index = -1
for index, filename in enumerate(os.listdir(output_folder)):
    if filename.endswith(".bin"):
        file_path = os.path.join(output_folder, filename)

        # 读取加密文件并反序列化
        with open(file_path, 'rb') as file:
            encrypted_image_data = file.read()
        enc_image = ts.bfv_vector_from(public_context, encrypted_image_data)

        # 计算相似度
        start_time = time.time()
        result1 = enc_query_image.dot(enc_image)
        dis_a = enc_query_image.dot(enc_query_image)
        dis_b = enc_image.dot(enc_image)

        distance = abs(result1._decrypt()[0]/math.sqrt(abs(dis_a._decrypt()[0])))
        distance = 1-distance/math.sqrt(abs(dis_b.decrypt()[0]))
        end_time = time.time()
        time1 = end_time - start_time
        time_list_match.append(time1)

        print(f"No.{index}'s similarity: {distance}")

        # 更新最大相似度和序号
        if abs(distance) < abs(max_similarity):
            max_similarity = distance
            max_similarity_index = index
            
print(f"compare cost: {sum(time_list_match)} seconds")
print(f"compare a single image cost: {np.mean(time_list_match)} seconds on average")

print("-----------------decrypt image(if matching is successful)---------------------")
if(abs(max_similarity) < 10**-6):
    print("Match successfully!")
    print(f"The most similar image index: {max_similarity_index}, similarity: {max_similarity}")

    # 从文件中读取并解密最相似的图片
    most_similar_image_path = f"{output_folder}/data_{max_similarity_index}.bin"

    with open(most_similar_image_path, 'rb') as file:
        encrypted_image_data = file.read()

    # 反序列化并解密
    enc_most_similar_image = ts.bfv_vector_from(public_context, encrypted_image_data)
    start_time = time.time()
    decrypt_ = enc_most_similar_image.decrypt()
    end_time = time.time()
    time1 = end_time - start_time
    print(f"decrypt cost: {time1} seconds")

    decrypted_image_array = np.array(decrypt_)
    decrypted_image_shape = (64, 64)  # 图像大小 64x64
    decrypted_image = decrypted_image_array.reshape(decrypted_image_shape).astype(np.uint8)

    # 将解密后的图像保存
    image = Image.fromarray(decrypted_image)
    image.save("decrypted_most_similar_image.png")
else:
    print("Match failure!")
    print(f"The most similar image index: {max_similarity_index}, similarity: {max_similarity}")