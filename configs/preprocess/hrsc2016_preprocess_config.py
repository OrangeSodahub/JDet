from numpy import source


type='HRSC2016'
source_dataset_path='/home/flowey/dataset/HRSC2016'

tasks=[
    dict(
        label='train',
        config=dict(
            images_path=source_dataset_path+'/Train/AllImages',
            xml_path=source_dataset_path+'/Train/Annotations',
            imageset_file=source_dataset_path+'/Train/train.txt',
            out_annotation_file=source_dataset_path+'/Train/labels.pkl',
        )
    ),
    dict(
        label='test',
        config=dict(
            images_path=source_dataset_path+'/Test/AllImages',
            xml_path=source_dataset_path+'/Test/Annotations',
            imageset_file=source_dataset_path+'/Test/test.txt',
            out_annotation_file=source_dataset_path+'/Test/labels.pkl',
        )
    )
]
