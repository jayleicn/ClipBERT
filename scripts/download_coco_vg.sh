# Download COCO+VG images and caption annotations
DOWNLOAD=$1

for FOLDER in 'vis_db' 'txt_db' 'pretrained' 'finetune'; do
    if [ ! -d $DOWNLOAD/$FOLDER ] ; then
        mkdir -p $DOWNLOAD/$FOLDER
    fi
done

BLOB='https://convaisharables.blob.core.windows.net/clipbert'

# vis dbs
for DATASET in 'vg' 'coco_test2015' 'coco_train2014_val2014'; do        
    if [ ! -d $DOWNLOAD/vis_db/$DATASET ] ; then
        wget -nc $BLOB/vis_db/$DATASET.tar -P $DOWNLOAD/vis_db/
        mkdir -p $DOWNLOAD/vis_db/$DATASET
        tar -xvf $DOWNLOAD/vis_db/$DATASET.tar -C $DOWNLOAD/vis_db/$DATASET
        rm $DOWNLOAD/vis_db/$DATASET.tar
    fi
done

# text dbs
if [ ! -d $DOWNLOAD/txt_db/pretrain/ ] ; then
    wget -nc $BLOB/txt_db/pretrain.tar -P $DOWNLOAD/txt_db/
    mkdir -p $DOWNLOAD/txt_db/pretrain
    tar -xvf $DOWNLOAD/txt_db/pretrain.tar -C $DOWNLOAD/txt_db/pretrain
    rm $DOWNLOAD/txt_db/pretrain.tar
fi

# pretraining init
# bert base 
if [ ! -d $DOWNLOAD/pretrained/bert-base-uncased ] ; then
    wget -nc $BLOB/pretrained/bert-base-uncased.tar -P $DOWNLOAD/pretrained/
    mkdir -p $DOWNLOAD/pretrained/bert-base-uncased
    tar -xvf $DOWNLOAD/pretrained/bert-base-uncased.tar -C $DOWNLOAD/pretrained/bert-base-uncased
    rm $DOWNLOAD/pretrained/bert-base-uncased.tar
fi

# grid-feat model
wget -nc $BLOB/pretrained/grid_feat_R-50.pth -O $DOWNLOAD/pretrained/grid_feat_R-50.pth