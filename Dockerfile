FROM pytorch/pytorch:latest

ARG TZ=Asia/Tokyo

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && apt update && apt install -y \
    git \
    wget \
    curl \
    unzip \
    vim \
    x11-apps \
    libxext6 \
    libx11-6 \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    freeglut3-dev \
    build-essential cmake libclang-dev \
    && apt autoremove -y \
    && apt clean -y

RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.2/zsh-in-docker.sh)" -- \
    -t robbyrussell

RUN mkdir /root/.vim
COPY .vimrc /root/.vimrc

WORKDIR /kaggle
COPY requirements.txt /kaggle
RUN pip install -r requirements.txt

RUN pip install lightgbm --install-option=--cuda

# install jupyter-vim-binding
RUN mkdir -p $(jupyter --data-dir)/nbextensions && \
    cd $(jupyter --data-dir)/nbextensions && \
    git clone https://github.com/lambdalisue/jupyter-vim-binding vim_binding && \
    jupyter nbextension enable vim_binding/vim_binding
