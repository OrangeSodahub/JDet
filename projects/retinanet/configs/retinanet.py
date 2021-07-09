model = dict(
    type = "RetinaNet",
    backbone = dict(
        type = "Resnet50_v1d",
        return_stages =  ["layer1","layer2","layer3","layer4"],
        pretrained = True),
    neck = dict(
        type= "FPN",
        in_channels= [256,512,1024,2048],
        out_channels= 256,
        start_level= 1,
        add_extra_convs= "on_output",
        num_outs= 5,
        upsample_cfg = dict(
            mode= "bilinear",
            tf_mode= True),
        upsample_div_factor= 2,
        relu_before_extra_convs= True),
    rpn_net = dict(
        type= "RetinaHead",
        n_class= 15,
        in_channels= 256,
        stacked_convs= 4,
        mode= "R",
        score_threshold= 0.05,
        nms_iou_threshold= 0.3,
        max_dets= 10000,

        anchor_generator = dict(
          type= "AnchorGeneratorYangXue",
          strides= [8, 16, 32, 64, 128],
          ratios= [1, 0.5, 2.0, 0.3333333333333333, 3.0, 5.0, 0.2],
          scales= [1, 1.2599210498948732, 1.5874010519681994],
          base_sizes= [32, 64, 128, 256, 512],
          angles= [-90, -75, -60, -45, -30, -15],
          mode= "H",
          yx_base_size= 4.)),
)
dataset = dict(
    # train=
    #     type= COCODataset
    #     root= /mnt/disk/lxl/dataset/coco/images/train2017
    #     anno_file= /mnt/disk/lxl/dataset/coco/annotations/instances_train2017.json
    #     batch_size= 16
    #     num_workers= 4
    #     shuffle= True
    #     filter_empty_gt= True
    #     transforms=
    #         - type= "Resize"
    #           min_size= [800]
    #           max_size= 1333
    #         - type= "RandomFlip"
    #           prob= 0.5
    #         - type= Pad
    #           size_divisor= 32
    #         - type= "Normalize"
    #           mean= [123.675, 116.28, 103.53]
    #           std= [58.395, 57.12, 57.375]
    # val=
    #     type= COCODataset
    #     root= /mnt/disk/lxl/dataset/coco/images/val2017
    #     anno_file= /mnt/disk/lxl/dataset/coco/annotations/instances_val2017.json
    #     batch_size= 2
    #     num_workers= 4
    #     filter_empty_gt= False
    #     transforms=
    #         - type= Pad
    #           size_divisor= 32
    #         - type= "Normalize"
    #           mean= [123.675, 116.28, 103.53]
    #           std= [58.395, 57.12, 57.375]
    test = dict(
      type= "ImageDataset",
      img_files= "/mnt/disk/cxjyxx_me/JAD/datasets/DOTA/splits/test_600_150/image_names.pkl",
      img_prefix= "/mnt/disk/cxjyxx_me/JAD/datasets/DOTA/splits/test_600_150/images/",
      # img_files= "/mnt/disk/cxjyxx_me/JAD/datasets/DOTA_mini/splits/test_600_150/image_names.pkl",
      # img_prefix= "/mnt/disk/cxjyxx_me/JAD/datasets/DOTA_mini/splits/test_600_150/images/",
      transforms= [
        dict(
          type= "RotatedResize",
          min_size= 800,
          max_size= 800),
        # dict(
        #   type= "Normalize",
        #   mean=  [0, 0, 0],
        #   std= [1, 1, 1],
        #   to_bgr= False)
        dict(
          type= "Normalize",
          mean=  [123.675, 116.28, 103.53],
          std= [58.395, 57.12, 57.375],
          to_bgr= False)
      ],
      num_workers= 4,
      batch_size= 32))
optim = dict(
    type= "SGD",
    lr= 0.01,
    momentum= 0.9,
    weight_decay= 0.0001)

scheduler = dict(
    type= "StepLR",
    warmup= "linear",
    warmup_iters= 500,
    warmup_ratio= 0.001,
    milestones= [8, 11])

logger = dict(
    type= "RunLogger")

work_dir = "./exp/retinanet"
max_epoch = 12
eval_interval = 1
log_interval = 50
checkpoint_interval = 1
