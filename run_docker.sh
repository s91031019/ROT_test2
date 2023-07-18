docker run -it -d --gpus all --shm-size 8G  --name yolox -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -v /home/rvl224/文件/y/YOLOX:/workspace/YOLOX -u 0 yolox /bin/bash
#docker cp /home/rvl224/文件/wilbur_data yolox:/data

#docker exec -it yolox /bin/bash

