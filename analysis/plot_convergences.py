if __name__ == '__main__':

    ######################################################
    # Choose a list of log to plot together for comparison

    ######################################################
    # My logs: choose the logs to show
    logs_names = ['name_log_1',
                  'name_log_2',
                  'name_log_3']

    logs = ['checkpoints/xxx/xxx', 'checkpoints/yyy/yyy', 'checkpoints/zzz/zzz']

    # 显示多个训练过程的  学习率，损失函数，时间曲线
    compare_trainings(logs, logs_names)

    if


    if config.dataset_task == 'classification':
        compare_convergences_classif(logs, logs_names)
    elif config.dataset_task == 'cloud_segmentation':
        if config.dataset.startswith('S3DIS'):
            dataset = S3DISDataset(config, load_data=False)
            compare_convergences_segment(dataset, logs, logs_names)
    elif config.dataset_task == 'slam_segmentation':
        if config.dataset.startswith('SemanticKitti'):
            dataset = SemanticKittiDataset(config)
            compare_convergences_SLAM(dataset, logs, logs_names)
    else:
        raise ValueError('Unsupported dataset : ' + plot_dataset)
