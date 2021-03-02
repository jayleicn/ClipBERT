# Download MSRVTT videos and annotations for Retrieval + QA + MC-Test
DOWNLOAD=$1

for FOLDER in 'vis_db' 'txt_db' 'pretrained' 'finetune'; do
    if [ ! -d $DOWNLOAD/$FOLDER ] ; then
        mkdir -p $DOWNLOAD/$FOLDER
    fi
done

BLOB='https://convaisharables.blob.core.windows.net/clipbert'

# vis dbs
if [ ! -d $DOWNLOAD/vis_db/msrvtt/ ] ; then
    wget -nc $BLOB/vis_db/msrvtt.tar -P $DOWNLOAD/vis_db/
    mkdir -p $DOWNLOAD/vis_db/msrvtt
    tar -xvf $DOWNLOAD/vis_db/msrvtt.tar -C $DOWNLOAD/vis_db/msrvtt
    rm $DOWNLOAD/vis_db/msrvtt.tar
fi

# text dbs
if [ ! -d $DOWNLOAD/txt_db/msrvtt_retrieval/ ] ; then
    # MC-Test is included
    wget -nc $BLOB/txt_db/msrvtt_retrieval.tar -P $DOWNLOAD/txt_db/
    mkdir -p $DOWNLOAD/txt_db/msrvtt_retrieval
    tar -xvf $DOWNLOAD/txt_db/msrvtt_retrieval.tar -C $DOWNLOAD/txt_db/msrvtt_retrieval
    rm $DOWNLOAD/txt_db/msrvtt_retrieval.tar
fi

if [ ! -d $DOWNLOAD/txt_db/msrvtt_qa/ ] ; then
    wget -nc $BLOB/txt_db/msrvtt_qa.tar -P $DOWNLOAD/txt_db/
    mkdir -p $DOWNLOAD/txt_db/msrvtt_qa
    tar -xvf $DOWNLOAD/txt_db/msrvtt_qa.tar -C $DOWNLOAD/txt_db/msrvtt_qa
    rm $DOWNLOAD/txt_db/msrvtt_qa.tar
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