# Download Models:
# 1, pretrained model
DOWNLOAD=$1

if [ ! -d $DOWNLOAD/pretrained ] ; then
    mkdir -p $DOWNLOAD/pretrained
fi

BLOB='https://convaisharables.blob.core.windows.net/clipbert'

# This will not overwrite model
wget -nc $BLOB/pretrained/clipbert_image_text_pretrained.pt -O $DOWNLOAD/pretrained/clipbert_image_text_pretrained.pt

# 2, pretrained BERT model
wget -nc $BLOB/pretrained/bert-base-uncased.tar -P $DOWNLOAD/pretrained/
mkdir -p $DOWNLOAD/pretrained/bert-base-uncased
tar -xvf $DOWNLOAD/pretrained/bert-base-uncased.tar -C $DOWNLOAD/pretrained/bert-base-uncased
rm $DOWNLOAD/pretrained/bert-base-uncased.tar

# 3, grid-feat model
wget -nc $BLOB/pretrained/grid_feat_R-50.pth -O $DOWNLOAD/pretrained/grid_feat_R-50.pth