base_dir = '/Users/rohit/Documents/LFVS/saver/checkpoint_saver/GPU'
categories = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
models = []
model_name_lr = []
for model in categories:
    #print ("category :{}".format(category))
    model_path = os.path.join(base_dir, model)
    all_lr = []
    name_lr = []
    for lr_dir in os.listdir(model_path):
        lr_path = os.path.join(model_path, lr_dir)
        if(os.path.isdir(lr_path)):
            name_lr.append(model + "_" + lr_dir)
            this_lr = []
            print ("***** {} ***".format(lr_path))
            for items in os.listdir(lr_path):
                if (items == "loss_train.npy"):
                    lr_loss_train_path = os.path.join(lr_path, items)
                    loss_train = (np.load(lr_loss_train_path))
                    print ("loss_train : {}".format(loss_train.shape))
                    this_lr.append(loss_train)
                elif (items == "loss.npy"):
                    lr_loss_test_path = os.path.join(lr_path, items)
                    loss_test = (np.load(lr_loss_test_path))
                    print ("loss_test : {}".format(loss_test.shape))
                    this_lr.append(loss_test)
            all_lr.append(this_lr)
    model_name_lr.append(name_lr)
    models.append(all_lr)

limit = 10000
model_name_lr = np.array(model_name_lr)
models = np.array(models)
x = np.linspace(1, limit, limit)
figsize = (20, 15)
cols = models.shape[0]
rows = models.shape[1]


gs = gridspec.GridSpec(rows, cols)
fig1 = plt.figure(num=1, figsize=figsize)

ax = []
for m in range(cols):
    for l in range(rows):
        ax.append(fig1.add_subplot(gs[l,m]))
        ax[-1].set_title(model_name_lr[m,l])
        loss_train = models[m,l,0]
        loss_test = models[m,l,1]
        #print ("***{}***".format(model_name_lr[m,l]))
        #print ("loss_train : {}".format(loss_train.shape))
        #print ("loss_test : {}".format(loss_test.shape))
        ax[-1].plot(x,loss_train[0:limit],x,loss_test[0:limit])
plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9,hspace=0.5)
plt.plot()
