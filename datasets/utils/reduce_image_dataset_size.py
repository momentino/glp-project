import json
import os

aro_folder = '/home/filippo/Downloads/aro/aro_images'

valse_folder = '/home/filippo/Downloads/VALSE'

def reduce_image_dataset_size_aro(image_list):
    with open('../ARO/transitive_visual_genome_relation.json', 'r') as f:
        data = json.load(f)
        for i in image_list:
            image_found = False
            for key, value in data.items():
                image = value['image_id']
                if(i[:-4] == image):
                    image_found = True
                print(image_found)
            if(image_found == False):
                os.remove(os.path.join(aro_folder,i))

def reduce_image_dataset_size_valse(image_list):
    with open('../VALSE/transitive_actant_swap.json', 'r') as f:
        data = json.load(f)
        for i in image_list:
            image_found = False
            for key, value in data.items():
                image = value['image_file']
                if(i == image):
                    image_found = True
                print(image_found)
            if(image_found == False):
                os.remove(os.path.join(valse_folder,i))



if __name__ == '__main__':
    """image_list = []
    for filename in os.listdir(aro_folder):
        file_path = os.path.join(aro_folder, filename)
        if os.path.isfile(file_path):
            # Do something with the file
            image_list.append(filename)
            print(f'Processing file: {filename}')
    reduce_image_dataset_size_aro(image_list)"""
    image_list = []
    for filename in os.listdir(valse_folder):
        file_path = os.path.join(valse_folder, filename)
        if os.path.isfile(file_path):
            # Do something with the file
            image_list.append(filename)
            print(f'Processing file: {filename}')
    reduce_image_dataset_size_valse(image_list)