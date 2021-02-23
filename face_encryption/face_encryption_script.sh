#may need to rewrite the path of `shape_predictor_68_face_landmarks.dat` in `../stylegan2-pytorch/alignFace.py` and `../FaceSwap/face_detection`  when making these code public.

img_directory=$1
orig_name=$2
swap_target_name=$3

#Encryption stage:

## 2. Find the latent vector
source deactivate 
source activate stylegan2

### 2.1 Crop the image under a certain format
python ../stylegan2-pytorch/alignFace.py --directory $img_directory --input_name $orig_name

### 2.2 white out the background (TODO: NEEDS TO BE REPLACED BY REAL CODES INSTEAD OF API)
python ../stylegan2-pytorch/rm_background.py --directory $img_directory --input_name ${orig_name}_cropped

### 2.3 Find the latent vector and get the projected images
python ../stylegan2-pytorch/projector.py --ckpt ../stylegan2-pytorch/stylegan2-ffhq-config-f.pt --step 1000 --size=1024 --output_folder $img_directory/ ${img_directory}/${orig_name}_cropped_no-bg.jpg
## 1. Generate swapped img

source deactivate
source activate faceswap
python ../FaceSwap/main.py --src ${img_directory}/${swap_target_name}.jpg --dst ${img_directory}/${orig_name}.jpg --out ${img_directory}/${orig_name}_encrypt_init.jpg --no_debug_window --correct_color

## 3. encrypt the latent vector into ${orig_name}_encrypt_init.jpg and get ${img_directory}/${orig_name}_encrypt.jpg (TODO, right now fake)
cp ${img_directory}/${orig_name}_encrypt_init.jpg ${img_directory}/${orig_name}_encrypt.jpg

#Decryption stage:
## 1. Decrypt the latent vector and generate and original face (TODO, right now fake)

## 2. Swap face back
source deactivate
source activate faceswap
python ../FaceSwap/main.py --src ${img_directory}/${orig_name}_cropped_no-bg-project.jpg --dst ${img_directory}/${orig_name}_encrypt.jpg --out ${img_directory}/${orig_name}_decrypt.jpg --no_debug_window --correct_color

#Get the cropped images
source deactivate 
source activate stylegan2
python ../stylegan2-pytorch/alignFace.py --directory $img_directory --input_name ${orig_name}_decrypt
python ../stylegan2-pytorch/alignFace.py --directory $img_directory --input_name ${orig_name}_encrypt