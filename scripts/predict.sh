#python ~/Pytorch-UNet/LargeImagePredictor.py 38-T-left-corridor-1
#python ~/Pytorch-UNet/LargeImagePredictor.py 38-T-left-corridor-2
#python ~/Pytorch-UNet/LargeImagePredictor.py 38-T-left-corridor-3
#python ~/Pytorch-UNet/LargeImagePredictor.py 38-T-left-corridor-4
#python ~/Pytorch-UNet/LargeImagePredictor.py 38-T-back-corridor-1
#python ~/Pytorch-UNet/LargeImagePredictor.py 38-T-back-corridor-2
#python ~/Pytorch-UNet/LargeImagePredictor.py 38-T-back-corridor-3
#python ~/Pytorch-UNet/LargeImagePredictor.py 38-T-back-corridor-4
#python ~/Pytorch-UNet/LargeImagePredictor.py 38-T-corbel-1
#python ~/Pytorch-UNet/LargeImagePredictor.py 38-T-corbel-2
#python ~/Pytorch-UNet/LargeImagePredictor.py 38-T-corbel-3
#python ~/Pytorch-UNet/LargeImagePredictor.py 38-T-corbel-4
#python ~/Pytorch-UNet/LargeImagePredictor.py 38-T-rightwall-1
#python ~/Pytorch-UNet/LargeImagePredictor.py 38-T-rightwall-2
#python ~/Pytorch-UNet/LargeImagePredictor.py 38-T-rightwall-3
#python ~/Pytorch-UNet/LargeImagePredictor.py 38-T-rightwall-4
#python ~/Pytorch-UNet/LargeImagePredictor.py 38-T-right-corridor-1
#python ~/Pytorch-UNet/LargeImagePredictor.py 38-T-right-corridor-2
#python ~/Pytorch-UNet/LargeImagePredictor.py 38-T-right-corridor-3
#python ~/Pytorch-UNet/LargeImagePredictor.py 38-T-right-corridor-4
#python ~/Pytorch-UNet/LargeImagePredictor.py 38-T-leftwall-1
#python ~/Pytorch-UNet/LargeImagePredictor.py 38-T-leftwall-2
#python ~/Pytorch-UNet/LargeImagePredictor.py 38-T-leftwall-3
#python ~/Pytorch-UNet/LargeImagePredictor.py 38-T-leftwall-4

python ~/Pytorch-UNet/LargeImagePredictor.py ~/RESTORATION/38-T-left-corridor.png ~/RESTORATION/predict/38-T-left-corridor-unet-type-p128.png
python ~/Pytorch-UNet/LargeImagePredictor.py ~/RESTORATION/downsampled/38-T-corbel_4xDownSampled.png ~/RESTORATION/predict/38-T-corbel-unet-type-p128.png
python ~/Pytorch-UNet/LargeImagePredictor.py ~/RESTORATION/downsampled/38-T-right-corridor_4xDownSampled.png ~/RESTORATION/predict/38-T-right-corridor-unet-type-p128.png
python ~/Pytorch-UNet/LargeImagePredictor.py ~/RESTORATION/downsampled/38-T-leftwall_4xDownSampled.png ~/RESTORATION/predict/38-T-leftwall-unet-type-p128.png