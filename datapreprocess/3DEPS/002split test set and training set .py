import os
import random
import shutil


source_folder = '/merge'
target_folder = '/train'   # trainset_dir or testset_dir


txt_files = [os.path.join(source_folder, f) for f in os.listdir(source_folder) if f.endswith('.txt')]

if not os.path.exists(target_folder):
    os.makedirs(target_folder)

all_files_number = 182
selected_files_number = 364   # train files number  (182 if test)
random_numbers = random.sample(range(all_files_number), selected_files_number)

for number in random_numbers:
    if number < len(txt_files):
        file_to_move = txt_files[number]
        file_name = os.path.basename(file_to_move)
        target_path = os.path.join(target_folder, file_name)

        # move files
        shutil.move(file_to_move, target_path)
        print(f"Moved {file_name} to {target_path}")

print("done")
