docker build -t grambank .
docker run -it --rm --name grambank -v $PWD:/src grambank /bin/bash