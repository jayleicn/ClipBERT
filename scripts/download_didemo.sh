# Download DiDeMo videos and annotations
DOWNLOAD=$1

for FOLDER in 'vis_db' 'txt_db' 'pretrained' 'finetune'; do
    if [ ! -d $DOWNLOAD/$FOLDER ] ; then
        mkdir -p $DOWNLOAD/$FOLDER
    fi
done

BLOB='https://convaisharables.blob.core.windows.net/clipbert'

# vis dbs
if [ ! -d $DOWNLOAD/vis_db/didemo/ ] ; then
    wget -nc $BLOB/vis_db/didemo.tar -P $DOWNLOAD/vis_db/
    mkdir -p $DOWNLOAD/vis_db/didemo
    tar -xvf $DOWNLOAD/vis_db/didemo.tar -C $DOWNLOAD/vis_db/didemo
    rm $DOWNLOAD/vis_db/didemo.tar
fi

# text dbs
if [ ! -d $DOWNLOAD/txt_db/didemo_retrieval/ ] ; then
    wget -nc $BLOB/txt_db/didemo_retrieval.tar -P $DOWNLOAD/txt_db/
    mkdir -p $DOWNLOAD/txt_db/didemo_retrieval
    tar -xvf $DOWNLOAD/txt_db/didemo_retrieval.tar -C $DOWNLOAD/txt_db/didemo_retrieval
    rm $DOWNLOAD/txt_db/didemo_retrieval.tar
fi

# pretrained
if [ ! -f $DOWNLOAD/pretrained/clipbert_image_text_pretrained.pt ] ; then
    wget -nc $BLOB/pretrained/clipbert_image_text_pretrained.pt -P $DOWNLOAD/pretrained/
fi

# bert base (needed for tokenization)
if [ ! -d $DOWNLOAD/pretrained/bert-base-uncased ] ; then
    wget -nc $BLOB/pretrained/bert-base-uncased.tar -P $DOWNLOAD/pretrained/
    mkdir -p $DOWNLOAD/pretrained/bert-base-uncased
    tar -xvf $DOWNLOAD/pretrained/bert-base-uncased.tar -C $DOWNLOAD/pretrained/bert-base-uncased
    rm $DOWNLOAD/pretrained/bert-base-uncased.tar
fi