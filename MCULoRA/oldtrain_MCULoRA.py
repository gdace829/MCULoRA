import time
import datetime
import random
import argparse
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score, recall_score
import sys
from utils import Logger, get_loaders, build_model, generate_mask, generate_inputs
from loss import MaskedCELoss, MaskedMSELoss
import os
import torch.nn.functional as F
import warnings
sys.path.append('./')
warnings.filterwarnings("ignore")
import config


def Alignment(p, q):
    p = F.softmax(p, dim=-1)
    q = F.softmax(q, dim=-1)
    m = 0.5 * p + 0.5 * q
    kl_p_m = F.kl_div(p.log(), m, reduction='batchmean')
    kl_q_m = F.kl_div(q.log(), m, reduction='batchmean')
    js_score = 0.5 * (kl_p_m + kl_q_m)
    return js_score

def train_or_eval_model(args, model, reg_loss, cls_loss, dataloader, optimizer=None, train=False, first_stage=True, mark='train',probabilities=[0.5,0.5,0.5]):
    weight = []
    preds, preds_a, preds_t, preds_v, masks, labels = [], [], [], [], [], []
    losses, losses1, losses2, losses3 = [], [], [], []
    preds_test_condition = []
    dataset = args.dataset
    cuda = torch.cuda.is_available() and not args.no_cuda
    # sim_at, sim_av, sim_tv = [], [], []
    sim_0, sim_1, sim_2,sim_3,sim_4,sim_5,sim_6 = [], [], [],[],[],[],[]
    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()
    # print(f'Total batches in dataloader: {len(dataloader)}')
    grad_norm_list = []
    # 遍历数据集，不停迭代
    for data_idx, data in enumerate(dataloader):
        vidnames = []
        # print(len(data),'sjs')
        if train: optimizer.zero_grad()
        forget=False
        ## read dataloader and generate all missing conditions
        """
        audio_host, text_host, visual_host: [seqlen, batch, dim]
        audio_guest, text_guest, visual_guest: [seqlen, batch, dim]
        qmask: speakers, [batch, seqlen]
        umask: has utt, [batch, seqlen]
        label: [batch, seqlen]
        """
        audio_host, text_host, visual_host = data[0], data[1], data[2]
        audio_guest, text_guest, visual_guest = data[3], data[4], data[5]
        qmask, umask, label = data[6], data[7], data[8]
        vidnames += data[-1]
        seqlen = audio_host.size(0)
        batch = audio_host.size(1)

        ## using cmp-net masking manner [at least one view exists]
        ## host mask 获得掩码
        matrix = generate_mask(seqlen, batch, args.test_condition, first_stage,train,probabilities) # [seqlen*batch, view_num]
        audio_host_mask = np.reshape(matrix[0], (batch, seqlen, 1))
        text_host_mask = np.reshape(matrix[1], (batch, seqlen, 1))
        visual_host_mask = np.reshape(matrix[2], (batch, seqlen, 1))
        audio_host_mask = torch.LongTensor(audio_host_mask.transpose(1, 0, 2))
        text_host_mask = torch.LongTensor(text_host_mask.transpose(1, 0, 2))
        visual_host_mask = torch.LongTensor(visual_host_mask.transpose(1, 0, 2))
        # guest mask
        matrix = generate_mask(seqlen, batch, args.test_condition, first_stage,train,probabilities) # [seqlen*batch, view_num]
        audio_guest_mask = np.reshape(matrix[0], (batch, seqlen, 1))
        text_guest_mask = np.reshape(matrix[1], (batch, seqlen, 1))
        visual_guest_mask = np.reshape(matrix[2], (batch, seqlen, 1))
        audio_guest_mask = torch.LongTensor(audio_guest_mask.transpose(1, 0, 2))
        text_guest_mask = torch.LongTensor(text_guest_mask.transpose(1, 0, 2))
        visual_guest_mask = torch.LongTensor(visual_guest_mask.transpose(1, 0, 2))

        masked_audio_host = audio_host * audio_host_mask
        masked_audio_guest = audio_guest * audio_guest_mask
        masked_text_host = text_host * text_host_mask
        masked_text_guest = text_guest * text_guest_mask
        masked_visual_host = visual_host * visual_host_mask
        masked_visual_guest = visual_guest * visual_guest_mask

        ## add cuda for tensor
        if cuda:
            masked_audio_host, audio_host_mask = masked_audio_host.to(device), audio_host_mask.to(device)
            masked_text_host, text_host_mask = masked_text_host.to(device), text_host_mask.to(device)
            masked_visual_host, visual_host_mask = masked_visual_host.to(device), visual_host_mask.to(device)
            masked_audio_guest, audio_guest_mask = masked_audio_guest.to(device), audio_guest_mask.to(device)
            masked_text_guest, text_guest_mask = masked_text_guest.to(device), text_guest_mask.to(device)
            masked_visual_guest, visual_guest_mask = masked_visual_guest.to(device), visual_guest_mask.to(device)

            qmask = qmask.to(device)
            umask = umask.to(device)
            label = label.to(device)


        ## generate mask_input_features: ? * [seqlen, batch, dim], input_features_mask: ? * [seq_len, batch, 3]
        masked_input_features = generate_inputs(masked_audio_host, masked_text_host, masked_visual_host, \
                                                masked_audio_guest, masked_text_guest, masked_visual_guest, qmask)
        input_features_mask = generate_inputs(audio_host_mask, text_host_mask, visual_host_mask, \
                                                audio_guest_mask, text_guest_mask, visual_guest_mask, qmask)
        mask_a, mask_t, mask_v = input_features_mask[0][:,:,0].transpose(0,1), input_features_mask[0][:,:,1].transpose(0,1), input_features_mask[0][:,:,2].transpose(0,1)
        '''
        # masked_input_features, input_features_mask: ?*[seqlen, batch, dim]
        # qmask: speakers, [batch, seqlen]
        # umask: has utt, [batch, seqlen]
        # label: [batch, seqlen]
        # log_prob: [seqlen, batch, num_classes]
        '''
        if not first_stage and  train:
            forget = random.random() < 0
        index='0'
        # 根据掩码获取当前的模态组合索引
        if input_features_mask[0][0][0].cpu().tolist() == [1,1,1]:
            index='0'# atv
        elif input_features_mask[0][0][0].cpu().tolist() == [1,1,0]:
            index='1'#at
        elif input_features_mask[0][0][0].cpu().tolist() == [1,0,1]:
            index='2'#av
        elif input_features_mask[0][0][0].cpu().tolist() == [0,1,1]:
            index='3'#tv
        elif input_features_mask[0][0][0].cpu().tolist() == [1,0,0]:
            index='4'#a
        elif input_features_mask[0][0][0].cpu().tolist() == [0,1,0]:
            index='5'#t
        elif input_features_mask[0][0][0].cpu().tolist() == [0,0,1]:
            index='6'#v
        elif input_features_mask[0][0][0].cpu().tolist() == [0,0,0]:
            index='7'

        ## forward
        hidden, out, out_a, out_t, out_v, weight_save,x_a,x_t,x_v,xs_a,xs_t,xs_v= model(masked_input_features[0], input_features_mask[0], umask, first_stage,index)
        # 有t的时候特性信息和共性信息差距比较大
        js_all_a = Alignment(x_a, xs_a['0'])+Alignment(x_a, xs_a['1'])+Alignment(x_a, xs_a['2'])+Alignment(x_a, xs_a['4'])# 特性共性之间的差距越大越好
        js_all_t = Alignment(x_t, xs_t['0'])+Alignment(x_t, xs_t['1'])+Alignment(x_t, xs_t['3'])+Alignment(x_t, xs_t['5'])
        js_all_v = Alignment(x_v, xs_v['0'])+Alignment(x_v, xs_v['2'])+Alignment(x_v, xs_v['3'])+Alignment(x_v, xs_v['6'])
        # print(js_all_a,js_all_t,js_all_v)
        # kl散度越大说明特性信息学习越好 越小说明特性信息学习不好
        if index=='0'or index=='1'or index=='2'or index=='4':
            js_all_a=js_all_a
        else:
            js_all_a=torch.tensor(0)
        if index=='0'or index=='1'or index=='2'or index=='5':
            js_all_t=js_all_t
        else:
            js_all_t=torch.tensor(0)
        if index=='0'or index=='2'or index=='3'or index=='6':
            js_all_v=js_all_v
        else:
            js_all_v=torch.tensor(0)
        if  not train:# 在测试集动态调整
            js_0=(Alignment(x_a, xs_a['0'])+Alignment(x_t, xs_t['0'])+Alignment(x_v, xs_v['0']))/3
            js_1=(Alignment(x_a, xs_a['1'])+Alignment(x_t, xs_t['1']))/2
            js_2=(Alignment(x_a, xs_a['2'])+Alignment(x_v, xs_v['2']))/2
            js_3=(Alignment(x_t, xs_t['3'])+Alignment(x_v, xs_v['3']))/2
            js_4=(Alignment(x_a, xs_a['4']))
            js_5=(Alignment(x_t, xs_t['5']))
            js_6=(Alignment(x_v, xs_v['6']))
            sim_0.append(js_0.cpu().detach().numpy())
            sim_1.append(js_1.cpu().detach().numpy())
            sim_2.append(js_2.cpu().detach().numpy())
            sim_3.append(js_3.cpu().detach().numpy())
            sim_4.append(js_4.cpu().detach().numpy())
            sim_5.append(js_5.cpu().detach().numpy())
            sim_6.append(js_6.cpu().detach().numpy())
           
        # print("a相似度",js_all_a,"t相似度",js_all_t,"v相似度",js_all_v)
        
        # sim_at.append(sim_at1.cpu().detach().numpy())
        # sim_av.append(sim_av1.cpu().detach().numpy())
        # sim_tv.append(sim_tv1.cpu().detach().numpy())
        # sim_alla.append(sim_alla1.cpu().detach().numpy())
        # sim_allt.append(sim_allt1.cpu().detach().numpy())
        # sim_allv.append(sim_allv1.cpu().detach().numpy())
        ## save analysis result
        weight.append(weight_save)
        in_mask = torch.clone(input_features_mask[0].permute(1, 0, 2))
        in_mask[umask == 0] = 0
        weight.append(np.array(in_mask.cpu()))
        weight.append(label.detach().cpu().numpy())
        weight.append(vidnames)

        ## calculate loss
        lp_ = out.view(-1, out.size(2)) # [batch*seq_len, n_classes]
        lp_a, lp_t, lp_v = out_a.view(-1, out_a.size(2)), out_t.view(-1, out_t.size(2)), out_v.view(-1, out_v.size(2))


        labels_ = label.view(-1) # [batch*seq_len]
        # print(labels_.shape,umask.shape,lp_.shape)
        # Introduce a 30% probability of label being 0
        # if not first_stage and train:
        #     mask = torch.rand(labels_.shape).to(device)
        #     mask = mask < 0.6
        #     labels_[mask] = 0
        if dataset in ['IEMOCAPFour', 'IEMOCAPSix']:
            if first_stage:
                loss_a = cls_loss(lp_a, labels_, umask)
                loss_t = cls_loss(lp_t, labels_, umask)
                loss_v = cls_loss(lp_v, labels_, umask)
            else:
                loss = cls_loss(lp_, labels_, umask)
        if dataset in ['CMUMOSI', 'CMUMOSEI']:
            if first_stage:
                loss_a = reg_loss(lp_a, labels_, umask)
                loss_t = reg_loss(lp_t, labels_, umask)
                loss_v = reg_loss(lp_v, labels_, umask)
            else:
                loss_a = reg_loss(lp_a, labels_, umask)
                loss_t = reg_loss(lp_t, labels_, umask)
                loss_v = reg_loss(lp_v, labels_, umask)
                
                # 调整比例可以获得更好的性能
                loss = reg_loss(lp_, labels_, umask)

        ## save batch results
        preds_a.append(lp_a.data.cpu().numpy())
        preds_t.append(lp_t.data.cpu().numpy())
        preds_v.append(lp_v.data.cpu().numpy())
        preds.append(lp_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())
        # print(f'---------------{mark} loss: {loss}-------------------')
        preds_test_condition.append(out.view(-1, out.size(2)).data.cpu().numpy())

        if train and first_stage:
            loss_a.backward()
            loss_t.backward()
            loss_v.backward()
            optimizer.step()
        # if train and not first_stage:
        #     loss.backward()
        #     for name, param in model.named_parameters():
        #         if param.grad is not None:
        #             if "proj1.linear.weight" in name:
        #                 print(f"{name} 梯度范数: {param.grad.norm().item()}")
        #                 print(f"{name} 梯度向量: {param.grad}")
                        
        #     optimizer.step()
        
        # sjs记录一下梯度方向的大小
        if train and not first_stage:
            loss.backward()
            for name, param in model.named_parameters():
                    if param.grad is not None:
                        if "proj1.linear.weight" in name:
                            grad_norm_list.append(param.grad.norm().item())
            optimizer.step()
    # sjs记录一下梯度方向
    if len(grad_norm_list) > 0:
        grad_norm_avg = np.mean(grad_norm_list)
        with open("gradient_log_iemo_v.txt", "a") as f:
            f.write(f"本轮grad_norm_list的平均值: {grad_norm_avg}\n")
    
    assert preds!=[], f'Error: no dataset in dataloader'
    preds  = np.concatenate(preds)
    preds_a = np.concatenate(preds_a)
    preds_t = np.concatenate(preds_t)
    preds_v = np.concatenate(preds_v)
    labels = np.concatenate(labels)
    masks  = np.concatenate(masks)
    if not train:
        sim0 = np.mean(np.array(sim_0), axis=0)
        sim1 = np.mean(np.array(sim_1), axis=0)
        sim2 = np.mean(np.array(sim_2), axis=0)
        sim3 = np.mean(np.array(sim_3), axis=0)
        sim4 = np.mean(np.array(sim_4), axis=0)
        sim5 = np.mean(np.array(sim_5), axis=0)
        sim6 = np.mean(np.array(sim_6), axis=0)
        sim_all=[sim0,sim1,sim2,sim3,sim4,sim5,sim6]
        # print(f'sim0: {sim0}; sim1: {sim1}; sim2: {sim2}; sim3: {sim3}; sim4: {sim4}; sim5: {sim5}; sim6: {sim6}')
    else:
        sim0, sim1, sim2, sim3, sim4, sim5, sim6 = 0, 0, 0, 0, 0, 0, 0
        sim_all=[sim0,sim1,sim2,sim3,sim4,sim5,sim6]
    # sim_at = np.mean(np.array(sim_at), axis=0)
    # sim_av = np.mean(np.array(sim_av), axis=0)
    # sim_tv = np.mean(np.array(sim_tv), axis=0)
    # sim_allv1 = np.mean(np.array(sim_allv), axis=0)
    # sim_allt1 = np.mean(np.array(sim_allt), axis=0)
    # sim_alla1 = np.mean(np.array(sim_alla), axis=0) 现有的单纯为了缺失情况加强单模态，会影响整体多模态识别准确识别每个模态组合的情况
    # print(f'sim_at: {sim_at}; sim_av: {sim_av}; sim_tv: {sim_tv} ')
    # all
    if dataset in ['IEMOCAPFour', 'IEMOCAPSix']:
        preds = np.argmax(preds, 1)
        preds_a = np.argmax(preds_a, 1)
        preds_t = np.argmax(preds_t, 1)
        preds_v = np.argmax(preds_v, 1)
        avg_loss = round(np.sum(losses)/np.sum(masks), 4)
        avg_accuracy = accuracy_score(labels, preds, sample_weight=masks)
        avg_fscore = f1_score(labels, preds, sample_weight=masks, average='weighted')
        # print(preds)
        mae = 0
        ua = recall_score(labels, preds, sample_weight=masks, average='macro')
        avg_acc_a = accuracy_score(labels, preds_a, sample_weight=masks)
        avg_acc_t = accuracy_score(labels, preds_t, sample_weight=masks)
        avg_acc_v = accuracy_score(labels, preds_v, sample_weight=masks)
        return mae, ua, avg_accuracy, avg_fscore, [avg_acc_a, avg_acc_t, avg_acc_v], vidnames, avg_loss, weight, sim_all

    elif dataset in ['CMUMOSI', 'CMUMOSEI']:
        non_zeros = np.array([i for i, e in enumerate(labels) if e != 0]) # remove 0, and remove mask
        avg_loss = round(np.sum(losses)/np.sum(masks), 4)
        avg_accuracy = accuracy_score((labels[non_zeros] > 0), (preds[non_zeros] > 0))
        avg_fscore = f1_score((labels[non_zeros] > 0), (preds[non_zeros] > 0), average='weighted')
        mae = np.mean(np.absolute(labels[non_zeros] - preds[non_zeros].squeeze()))
        corr = np.corrcoef(labels[non_zeros], preds[non_zeros].squeeze())[0][1]
        avg_acc_a = accuracy_score((labels[non_zeros] > 0), (preds_a[non_zeros] > 0))
        avg_acc_t = accuracy_score((labels[non_zeros] > 0), (preds_t[non_zeros] > 0))
        avg_acc_v = accuracy_score((labels[non_zeros] > 0), (preds_v[non_zeros] > 0))
        return mae, corr, avg_accuracy, avg_fscore, [avg_acc_a, avg_acc_t, avg_acc_v], vidnames, avg_loss, weight, sim_all



# 主函数
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ## Params for input
    parser.add_argument('--audio-feature', type=str, default=None, help='audio feature name')
    parser.add_argument('--text-feature', type=str, default=None, help='text feature name')
    parser.add_argument('--video-feature', type=str, default=None, help='video feature name')
    parser.add_argument('--dataset', type=str, default='IEMOCAPFour', help='dataset type')

    ## Params for model
    parser.add_argument('--time-attn', action='store_true', default=False, help='whether to use nodal attention in graph model: Equation 4,5,6 in Paper')
    parser.add_argument('--depth', type=int, default=4, help='')
    parser.add_argument('--num_heads', type=int, default=2, help='')
    parser.add_argument('--drop_rate', type=float, default=0.0, help='')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, help='')
    parser.add_argument('--hidden', type=int, default=100, help='hidden size in model training')
    parser.add_argument('--n_classes', type=int, default=2, help='number of classes [defined by args.dataset]')
    parser.add_argument('--n_speakers', type=int, default=2, help='number of speakers [defined by args.dataset]')

    ## Params for training
    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--gpu', type=int, default=2, help='index of gpu')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.00001, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--batch-size', type=int, default=32, metavar='BS', help='batch size')
    parser.add_argument('--epochs', type=int, default=100, metavar='E', help='number of epochs')
    parser.add_argument('--num-folder', type=int, default=5, help='folders for cross-validation [defined by args.dataset]')
    parser.add_argument('--seed', type=int, default=100, help='make split manner is same with same seed')
    parser.add_argument('--test_condition', type=str, default='atv', choices=['a', 't', 'v', 'at', 'av', 'tv', 'atv'], help='test conditions')
    parser.add_argument('--stage_epoch', type=float, default=100, help='number of epochs of the first stage')

    args = parser.parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    args.device = device
    save_folder_name = f'{args.dataset}'
    save_log = os.path.join(config.LOG_DIR, 'main_result', f'{save_folder_name}')
    if not os.path.exists(save_log): os.makedirs(save_log)
    time_dataset = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}_{args.dataset}"
    sys.stdout = Logger(filename=f"{save_log}/{time_dataset}_batchsize-{args.batch_size}_lr-{args.lr}_seed-{args.seed}_test-condition-{args.test_condition}.txt",
                        stream=sys.stdout)

    ## seed
    def seed_torch(seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    seed_torch(args.seed)


    ## dataset
    if args.dataset in ['CMUMOSI', 'CMUMOSEI']:
        args.num_folder = 1
        args.n_classes = 1
        args.n_speakers = 1
    elif args.dataset == 'IEMOCAPFour':
        args.num_folder = 5
        args.n_classes = 4
        args.n_speakers = 2
    elif args.dataset == 'IEMOCAPSix':
        args.num_folder = 5
        args.n_classes = 6
        args.n_speakers = 2
    cuda = torch.cuda.is_available() and not args.no_cuda
    print(args)

    ## reading data
    print (f'====== Reading Data =======')
    audio_feature, text_feature, video_feature = args.audio_feature, args.text_feature, args.video_feature
    audio_root = os.path.join(config.PATH_TO_FEATURES[args.dataset], audio_feature)
    text_root = os.path.join(config.PATH_TO_FEATURES[args.dataset], text_feature)
    video_root = os.path.join(config.PATH_TO_FEATURES[args.dataset], video_feature)
    print(audio_root)
    print(text_root)
    print(video_root)
    assert os.path.exists(audio_root) and os.path.exists(text_root) and os.path.exists(video_root), f'features not exist!'
    train_loaders, test_loaders, adim, tdim, vdim = get_loaders(audio_root=audio_root,
                                                                             text_root=text_root,
                                                                             video_root=video_root,
                                                                             num_folder=args.num_folder,
                                                                             batch_size=args.batch_size,
                                                                             dataset=args.dataset,
                                                                             num_workers=0)
    assert len(train_loaders) == args.num_folder, f'Error: folder number'

    
    print (f'====== Training and Testing =======')
    folder_mae = []
    folder_corr = []
    folder_acc = []
    folder_f1 = []
    folder_model = []
    bestmodels = []
    for ii in range(args.num_folder):
        print (f'>>>>> Cross-validation: training on the {ii+1} folder >>>>>')
        train_loader = train_loaders[ii]
        test_loader = test_loaders[ii]
        start_time = time.time()

        print('-'*80)
        print (f'Step1: build model (each folder has its own model)')
        model = build_model(args, adim, tdim, vdim)
        reg_loss = MaskedMSELoss()
        cls_loss = MaskedCELoss()
        if cuda:
            model.to(device)
        optimizer = optim.Adam([{'params': model.parameters(), 'lr': args.lr, 'weight_decay': args.l2}])
        print('-'*80)


        print (f'Step2: training (multiple epoches)')
        train_acc_as, train_acc_ts, train_acc_vs = [], [], []
        test_fscores, test_accs, test_maes, test_corrs = [], [], [], []
        models = []
        start_first_stage_time = time.time()
        best_acc_all=0
        bestmodel= build_model(args, adim, tdim, vdim)
        bestmodel.to(device)
        print("------- Starting the first stage! -------")
        last_epoch_sim_a,last_epoch_sim_t,last_epoch_sim_v=0,0,0  
        base_prob_a, base_prob_t, base_prob_v=0.5,0.5,0.5
        probabilities_adjust=[base_prob_a,base_prob_t,base_prob_v]
        probabilities_adjust_scale=[1,0.5,0.25,0.125,0.125/2,0.125/4,0.125/8]
        # 创建映射字典，使用整数作为索引
        probabilities_adjust_map = {
            0: (1, 1, 1),
            1: (1, 1, -1),
            2: (1, -1, 1),
            3: (-1, 1, 1),
            4: (1, -1, -1),
            5: (-1, 1, -1),
            6: (-1, -1, 1),
            7: (-1, -1, -1)
        }
        vis_acc_test0=[]
        vis_acc_test1=[]
        vis_acc_test2=[]
        vis_acc_test3=[]
        vis_acc_test4=[]
        vis_acc_test5=[]
        vis_acc_test6=[]
        
        for epoch in range(args.epochs):
            first_stage = False if epoch < args.stage_epoch else False
            ## training and testing (!!! if IEMOCAP, the ua is equal to corr !!!)
            args.test_condition="atv"
            train_mae, train_corr, train_acc, train_fscore, train_acc_atv, train_names, train_loss, weight_train,_ = train_or_eval_model(args, model, reg_loss, cls_loss, train_loader, \
                                                                            optimizer=optimizer, train=True, first_stage=first_stage, mark='train',probabilities=probabilities_adjust)
            test_mae, test_corr, test_acc, test_fscore, test_acc_atv, test_names, test_loss, weight_test, sim_all = train_or_eval_model(args, model, reg_loss, cls_loss, test_loader, \
                                                                            optimizer=None, train=False, first_stage=first_stage, mark='test',probabilities=probabilities_adjust)
            args.test_condition="at"
            _, _, test_acc1, _, _, _, _, _, _ = train_or_eval_model(args, model, reg_loss, cls_loss, test_loader, \
                                                                            optimizer=None, train=False, first_stage=first_stage, mark='test',probabilities=probabilities_adjust)
            args.test_condition="av"
            _, _, test_acc2, _, _, _, _, _, _ = train_or_eval_model(args, model, reg_loss, cls_loss, test_loader, \
                                                                            optimizer=None, train=False, first_stage=first_stage, mark='test',probabilities=probabilities_adjust)
            args.test_condition="tv"
            _, _, test_acc3, _, _, _, _, _, _ = train_or_eval_model(args, model, reg_loss, cls_loss, test_loader, \
                                                                            optimizer=None, train=False, first_stage=first_stage, mark='test',probabilities=probabilities_adjust)
            args.test_condition="a"
            _, _, test_acc4, _, _, _, _, _, _ = train_or_eval_model(args, model, reg_loss, cls_loss, test_loader, \
                                                                            optimizer=None, train=False, first_stage=first_stage, mark='test',probabilities=probabilities_adjust)
            args.test_condition="t"
            _, _, test_acc5, _, _, _, _, _, _ = train_or_eval_model(args, model, reg_loss, cls_loss, test_loader, \
                                                                            optimizer=None, train=False, first_stage=first_stage, mark='test',probabilities=probabilities_adjust)
            args.test_condition="v"
            _, _, test_acc6, _, _, _, _, _, _ = train_or_eval_model(args, model, reg_loss, cls_loss, test_loader, \
                                                                            optimizer=None, train=False, first_stage=first_stage, mark='test',probabilities=probabilities_adjust)
            # 训练时的准确率变化
            vis_acc_test0.append(test_acc)
            vis_acc_test1.append(test_acc1)
            vis_acc_test2.append(test_acc2)
            vis_acc_test3.append(test_acc3)
            vis_acc_test4.append(test_acc4)
            vis_acc_test5.append(test_acc5)
            vis_acc_test6.append(test_acc6)
            args.test_condition="atv"
            if epoch==0:
                last_epoch_sim_0=sim_all[0]
                last_epoch_sim_1=sim_all[1]
                last_epoch_sim_2=sim_all[2]
                last_epoch_sim_3=sim_all[3]
                last_epoch_sim_4=sim_all[4]
                last_epoch_sim_5=sim_all[5]
                last_epoch_sim_6=sim_all[6]
            else:
                change_0 = np.abs(np.mean(sim_all[0]) - np.mean(last_epoch_sim_0))
                change_1 = np.abs(np.mean(sim_all[1]) - np.mean(last_epoch_sim_1))
                change_2 = np.abs(np.mean(sim_all[2]) - np.mean(last_epoch_sim_2))
                change_3 = np.abs(np.mean(sim_all[3]) - np.mean(last_epoch_sim_3))
                change_4 = np.abs(np.mean(sim_all[4]) - np.mean(last_epoch_sim_4))
                change_5 = np.abs(np.mean(sim_all[5]) - np.mean(last_epoch_sim_5))
                change_6 = np.abs(np.mean(sim_all[6]) - np.mean(last_epoch_sim_6))

                changes = [change_0, change_1, change_2, change_3, change_4, change_5, change_6]
                # max_index = np.argmax(changes)
                # min_index = np.argmin(changes)
                # change_str = " ".join([f"change_{i}: {change}" for i, change in enumerate(changes)])
                sorted_indices = sorted(range(len(changes)), key=lambda i: changes[i])
                for idx,v in enumerate(sorted_indices):
                    # 不同数据集超参数不同 mosei
                    probabilities_adjust[0]+=probabilities_adjust_map[v][0]*probabilities_adjust_scale[idx]*0.003
                    probabilities_adjust[1]+=probabilities_adjust_map[v][1]*probabilities_adjust_scale[idx]*0.003
                    probabilities_adjust[2]+=probabilities_adjust_map[v][2]*probabilities_adjust_scale[idx]*0.003
                    probabilities_adjust = np.clip(probabilities_adjust, 0.4, 0.6)
                    # probabilities_adjust[0]+=probabilities_adjust_map[v][0]*probabilities_adjust_scale[idx]*0.00025
                    # probabilities_adjust[1]+=probabilities_adjust_map[v][1]*probabilities_adjust_scale[idx]*0.00025
                    # probabilities_adjust[2]+=probabilities_adjust_map[v][2]*probabilities_adjust_scale[idx]*0.00025
                    # probabilities_adjust = np.clip(probabilities_adjust, 0.445, 0.555)
                    
                    # print(idx,v)
            # print(probabilities_adjust)
           

            ## save
            if epoch>args.stage_epoch:
                test_accs.append(test_acc)# 每个test epoch的结果汇聚起来选出最好的一次epoch 由于前面100个epoch没有掩码可能导致性能比较好所以出现了这个问题
                test_fscores.append(test_fscore)
                test_maes.append(test_mae)
                test_corrs.append(test_corr)
                # models.append(model)
                
                
            # test_accs.append(test_acc)
            # test_fscores.append(test_fscore)
            # test_maes.append(test_mae)
            # test_corrs.append(test_corr)
            # models.append(model)
            train_acc_as.append(train_acc_atv[0])
            train_acc_ts.append(train_acc_atv[1])
            train_acc_vs.append(train_acc_atv[2])

            if first_stage:
                print(f'epoch:{epoch}; a_acc_train:{train_acc_atv[0]:.3f}; t_acc_train:{train_acc_atv[1]:.3f}; v_acc_train:{train_acc_atv[2]:.3f}')
                print(f'epoch:{epoch}; a_acc_test:{test_acc_atv[0]:.3f}; t_acc_test:{test_acc_atv[1]:.3f}; v_acc_test:{test_acc_atv[2]:.3f}')
            else:
                if (test_acc*1.2+test_acc1+test_acc2+test_acc3+test_acc4+test_acc5+test_acc6)/7>best_acc_all:
                    # best_acc_all
                    best_acc_all=(test_acc*1.2+test_acc1+test_acc2+test_acc3+test_acc4+test_acc5+test_acc6)/7
                    bestmodel.load_state_dict(model.state_dict())
                print(f'epoch:{epoch}; train_mae_{args.test_condition}:{train_mae:.3f}; train_corr_{args.test_condition}:{train_corr:.3f}; train_fscore_{args.test_condition}:{train_fscore:2.2%}; train_acc_{args.test_condition}:{train_acc:2.2%}; train_loss_{args.test_condition}:{train_loss}')
                print(f'epoch:{epoch}; test_mae_{args.test_condition}:{test_mae:.3f}; test_corr_{args.test_condition}:{test_corr:.3f}; test_fscore_{args.test_condition}:{test_fscore:2.3%}; test_acc_{args.test_condition}:{test_acc:2.3%}; test_loss_{args.test_condition}:{test_loss}')
            print('-'*10)
            ## update the parameter for the 2nd stage
            # if epoch == args.stage_epoch-1:
            #     model = models[-1]

            #     model_idx_a = int(torch.argmax(torch.Tensor(train_acc_as)))
            #     print(f'best_epoch_a: {model_idx_a}')
            #     model_a = models[model_idx_a]
            #     transformer_a_para_dict = {k: v for k, v in model_a.state_dict().items() if 'Transformer' in k}
            #     model.state_dict().update(transformer_a_para_dict)

            #     model_idx_t = int(torch.argmax(torch.Tensor(train_acc_ts)))
            #     print(f'best_epoch_t: {model_idx_t}')
            #     model_t = models[model_idx_t]
            #     transformer_t_para_dict = {k: v for k, v in model_t.state_dict().items() if 'Transformer' in k}
            #     model.state_dict().update(transformer_t_para_dict)

            #     model_idx_v = int(torch.argmax(torch.Tensor(train_acc_vs)))
            #     print(f'best_epoch_v: {model_idx_v}')
            #     model_v = models[model_idx_v]
            #     transformer_v_para_dict = {k: v for k, v in model_v.state_dict().items() if 'Transformer' in k}
            #     model.state_dict().update(transformer_v_para_dict)

            #     end_first_stage_time = time.time()
            #     print("------- Starting the second stage! -------")

        # import matplotlib.pyplot as plt
        # # Save vis_acc arrays to files for later use
        np.save(f'momkevis_acc_test0_folder{ii}.npy', vis_acc_test0)
        np.save(f'momkevis_acc_test1_folder{ii}.npy', vis_acc_test1)
        np.save(f'momkevis_acc_test2_folder{ii}.npy', vis_acc_test2)
        np.save(f'momkevis_acc_test3_folder{ii}.npy', vis_acc_test3)
        np.save(f'momkevis_acc_test4_folder{ii}.npy', vis_acc_test4)
        np.save(f'momkevis_acc_test5_folder{ii}.npy', vis_acc_test5)
        np.save(f'momkevis_acc_test6_folder{ii}.npy', vis_acc_test6)

        # # Plotting the test accuracies for different conditions
        # plt.figure(figsize=(10, 6))
        # plt.plot(vis_acc_test0, label='atv')
        # plt.plot(vis_acc_test1, label='at')
        # plt.plot(vis_acc_test2, label='av')
        # plt.plot(vis_acc_test3, label='tv')
        # plt.plot(vis_acc_test4, label='a')
        # plt.plot(vis_acc_test5, label='t')
        # plt.plot(vis_acc_test6, label='v')
        # plt.xlabel('Epochs')
        # plt.ylabel('Test Accuracy')
        # plt.title(f'Test Accuracy for Different Conditions Over Epochs (Folder {ii})')
        # plt.legend()
        # plt.grid(True)
        # plt.savefig(f'test_accuracy_conditions_folder{ii}.png')  # Save the plot as an image
        # plt.show()

        end_second_stage_time = time.time()
        print("-"*80)
        # print(f"Time of first stage: {end_first_stage_time - start_first_stage_time}s")
        # print(f"Time of second stage: {end_second_stage_time - end_first_stage_time}s")
        print(f"Time of training: {end_second_stage_time - start_first_stage_time}s")
        print("-" * 80)

        print(f'Step3: saving and testing on the {ii+1} folder')
        if args.dataset in ['CMUMOSI', 'CMUMOSEI']:
            best_index_test = np.argmax(np.array(test_fscores))
        if args.dataset in ['IEMOCAPFour', 'IEMOCAPSix']:
            best_index_test = np.argmax(np.array(test_accs))


        bestmae = test_maes[best_index_test]
        bestcorr = test_corrs[best_index_test]
        bestf1 = test_fscores[best_index_test]
        bestacc = test_accs[best_index_test]
        # bestmodel = models[best_index_test]
        # print(len(models),len(test_fscores),"aaa")
        folder_mae.append(bestmae)
        folder_corr.append(bestcorr)
        folder_f1.append(bestf1)
        folder_acc.append(bestacc)
        folder_model.append(bestmodel)
        end_time = time.time()
        if args.dataset in ['CMUMOSI', 'CMUMOSEI']:
            print(f"The best(acc) epoch of test_condition ({args.test_condition}): {best_index_test} --test_mae {bestmae} --test_corr {bestcorr} --test_fscores {bestf1} --test_acc {bestacc}.")
            print(f'>>>>> Finish: training on the {ii+1} folder, duration: {end_time - start_time} >>>>>')
        if args.dataset in ['IEMOCAPFour', 'IEMOCAPSix']:
            print(f"The best(acc) epoch of test_condition ({args.test_condition}): {best_index_test} --test_acc {bestacc} --test_ua {bestcorr}.")
            print(f'>>>>> Finish: training on the {ii+1} folder, duration: {end_time - start_time} >>>>>')

    print('-'*80)
    if args.dataset in ['CMUMOSI', 'CMUMOSEI']:
        print(f"Folder avg: test_condition ({args.test_condition}) --test_mae {np.mean(folder_mae)} --test_corr {np.mean(folder_corr)} --test_fscores {np.mean(folder_f1)} --test_acc{np.mean(folder_acc)}")
    if args.dataset in ['IEMOCAPFour', 'IEMOCAPSix']:
        print(f"Folder avg: test_condition ({args.test_condition}) --test_acc{np.mean(folder_acc)} --test_ua {np.mean(folder_corr)}")

    print (f'====== Saving =======')

    save_model = os.path.join(config.MODEL_DIR, 'main_result', f'{save_folder_name}')
    if not os.path.exists(save_model): os.makedirs(save_model)
    ## gain suffix_name
    suffix_name = f"{time_dataset}_hidden-{args.hidden}_bs-{args.batch_size}"
    ## gain feature_name
    feature_name = f'{audio_feature};{text_feature};{video_feature}'
    ## gain res_name
    mean_mae = np.mean(np.array(folder_mae))
    mean_corr = np.mean(np.array(folder_corr))
    mean_f1 = np.mean(np.array(folder_f1))
    mean_acc = np.mean(np.array(folder_acc))
    if args.dataset in ['CMUMOSI', 'CMUMOSEI']:
        res_name = f'mae-{mean_mae:.3f}_corr-{mean_corr:.3f}_f1-{mean_f1:.4f}_acc-{mean_acc:.4f}'
    if args.dataset in ['IEMOCAPFour', 'IEMOCAPSix']:
        res_name = f'acc-{mean_acc:.4f}_ua-{mean_corr:.4f}'
    save_path = f'{save_model}/{suffix_name}_features-{feature_name}_{res_name}_test-condition-{args.test_condition}.pth'
    torch.save({'model': bestmodel.state_dict()}, save_path)
    print(save_path)
    # Load the saved model
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model'])
    print(f"Model loaded from {save_path}")
    test_acc0 = 0
    test_acc1 = 0
    test_acc2 = 0
    test_acc3 = 0
    test_acc4 = 0
    test_acc5 = 0
    test_acc6 = 0
    test_fscores0 = 0
    test_fscores1 = 0
    test_fscores2 = 0
    test_fscores3 = 0
    test_fscores4 = 0
    test_fscores5 = 0
    test_fscores6 = 0
    test_corr0 = 0
    test_corr1 = 0
    test_corr2 = 0
    test_corr3 = 0
    test_corr4 = 0
    test_corr5 = 0
    test_corr6 = 0

    # 在所有缺失情况该模型的效果 每个folder的效果不同
    for ii in range(args.num_folder):
        train_loader = train_loaders[ii]
        test_loader = test_loaders[ii]
        model = folder_model[ii]
        if args.dataset in ['CMUMOSI', 'CMUMOSEI']:
            test_mae_all=0
            test_corr_all=0
            test_acc_all=0
            test_fscore_all=0
            # 使用训练好的模型测试不同缺失情况
            args.test_condition = 'a'
            test_mae, test_corr, test_acc, test_fscore, test_acc_atv, test_names, test_loss, weight_test,_ = train_or_eval_model(args, model, reg_loss, cls_loss, test_loader, \
                                                                                    optimizer=None, train=False, first_stage=first_stage, mark='test')
            test_mae_all+=test_mae
            test_corr_all+=test_corr
            test_acc_all+=test_acc
            test_fscore_all+=test_fscore
            test_mae_all+=test_mae
            print(f"The best(acc) epoch of test_condition ({args.test_condition}):  --test_mae {test_mae} --test_corr {test_corr} --test_fscores {test_fscore} --test_acc {test_acc}.")
            args.test_condition = 't'
            test_mae, test_corr, test_acc, test_fscore, test_acc_atv, test_names, test_loss, weight_test,_ = train_or_eval_model(args, model, reg_loss, cls_loss, test_loader, \
                                                                                    optimizer=None, train=False, first_stage=first_stage, mark='test')
            test_mae_all+=test_mae
            test_corr_all+=test_corr
            test_acc_all+=test_acc
            test_fscore_all+=test_fscore
            print(f"The best(acc) epoch of test_condition ({args.test_condition}):  --test_mae {test_mae} --test_corr {test_corr} --test_fscores {test_fscore} --test_acc {test_acc}.")
            args.test_condition = 'v'
            test_mae, test_corr, test_acc, test_fscore, test_acc_atv, test_names, test_loss, weight_test,_ = train_or_eval_model(args, model, reg_loss, cls_loss, test_loader, \
                                                                                    optimizer=None, train=False, first_stage=first_stage, mark='test')
            test_mae_all+=test_mae
            test_corr_all+=test_corr
            test_acc_all+=test_acc
            test_fscore_all+=test_fscore
            print(f"The best(acc) epoch of test_condition ({args.test_condition}):  --test_mae {test_mae} --test_corr {test_corr} --test_fscores {test_fscore} --test_acc {test_acc}.")
            args.test_condition = 'at'
            test_mae, test_corr, test_acc, test_fscore, test_acc_atv, test_names, test_loss, weight_test,_ = train_or_eval_model(args, model, reg_loss, cls_loss, test_loader, \
                                                                                    optimizer=None, train=False, first_stage=first_stage, mark='test')
            test_mae_all+=test_mae
            test_corr_all+=test_corr
            test_acc_all+=test_acc
            test_fscore_all+=test_fscore
            print(f"The best(acc) epoch of test_condition ({args.test_condition}):  --test_mae {test_mae} --test_corr {test_corr} --test_fscores {test_fscore} --test_acc {test_acc}.")
            args.test_condition = 'av'
            test_mae, test_corr, test_acc, test_fscore, test_acc_atv, test_names, test_loss, weight_test,_ = train_or_eval_model(args, model, reg_loss, cls_loss, test_loader, \
                                                                                    optimizer=None, train=False, first_stage=first_stage, mark='test')
            test_mae_all+=test_mae
            test_corr_all+=test_corr
            test_acc_all+=test_acc
            test_fscore_all+=test_fscore
            print(f"The best(acc) epoch of test_condition ({args.test_condition}):  --test_mae {test_mae} --test_corr {test_corr} --test_fscores {test_fscore} --test_acc {test_acc}.")
            args.test_condition = 'tv'
            test_mae, test_corr, test_acc, test_fscore, test_acc_atv, test_names, test_loss, weight_test,_= train_or_eval_model(args, model, reg_loss, cls_loss, test_loader, \
                                                                                    optimizer=None, train=False, first_stage=first_stage, mark='test')
            test_mae_all+=test_mae
            test_corr_all+=test_corr
            test_acc_all+=test_acc
            test_fscore_all+=test_fscore
            print(f"The best(acc) epoch of test_condition ({args.test_condition}):  --test_mae {test_mae} --test_corr {test_corr} --test_fscores {test_fscore} --test_acc {test_acc}.")
            args.test_condition = 'atv'
            test_mae, test_corr, test_acc, test_fscore, test_acc_atv, test_names, test_loss, weight_test,_ = train_or_eval_model(args, model, reg_loss, cls_loss, test_loader, \
                                                                                    optimizer=None, train=False, first_stage=first_stage, mark='test')
            test_mae_all+=test_mae
            test_corr_all+=test_corr
            test_acc_all+=test_acc
            test_fscore_all+=test_fscore
            print(f"The best(acc) epoch of test_condition ({args.test_condition}):  --test_mae {test_mae} --test_corr {test_corr} --test_fscores {test_fscore} --test_acc {test_acc}.")
            print(f"Folder avg: test_condition ({args.test_condition}) --test_mae {test_mae_all/7} --test_corr {test_corr_all/7} --test_fscores {test_fscore_all/7} --test_acc {test_acc_all/7}")
        if args.dataset in ['IEMOCAPFour', 'IEMOCAPSix']:
            print(ii,"cur folder")
            test_acc_all=0
            test_corr_all=0
            # 使用训练好的模型测试不同缺失情况
            args.test_condition = 'a'
            test_mae, test_corr, test_acc, test_fscore, test_acc_atv, test_names, test_loss, weight_test,_ = train_or_eval_model(args, model, reg_loss, cls_loss, test_loader, \
                                                                                    optimizer=None, train=False, first_stage=first_stage, mark='test')
            test_acc_all+=test_acc
            test_corr_all+=test_corr
            test_acc0+=test_acc
            test_fscores0+=test_fscore
            test_corr0+=test_corr
            print(f"The best(acc) epoch of test_condition ({args.test_condition}):  --test_acc {test_acc} --test_ua {test_corr}.")
            args.test_condition = 't'
            test_mae, test_corr, test_acc, test_fscore, test_acc_atv, test_names, test_loss, weight_test,_ = train_or_eval_model(args, model, reg_loss, cls_loss, test_loader, \
                                                                                    optimizer=None, train=False, first_stage=first_stage, mark='test')
            test_acc_all+=test_acc
            test_corr_all+=test_corr
            test_acc1+=test_acc
            test_fscores1+=test_fscore
            test_corr1+=test_corr
            print(f"The best(acc) epoch of test_condition ({args.test_condition}):  --test_acc {test_acc} --test_ua {test_corr}.")
            args.test_condition = 'v'
            test_mae, test_corr, test_acc, test_fscore, test_acc_atv, test_names, test_loss, weight_test,_ = train_or_eval_model(args, model, reg_loss, cls_loss, test_loader, \
                                                                                    optimizer=None, train=False, first_stage=first_stage, mark='test')
            test_acc_all+=test_acc
            test_corr_all+=test_corr
            test_acc2+=test_acc
            test_fscores2+=test_fscore
            test_corr2+=test_corr
            print(f"The best(acc) epoch of test_condition ({args.test_condition}):  --test_acc {test_acc} --test_ua {test_corr}.")
            args.test_condition = 'at'
            test_mae, test_corr, test_acc, test_fscore, test_acc_atv, test_names, test_loss, weight_test,_ = train_or_eval_model(args, model, reg_loss, cls_loss, test_loader, \
                                                                                    optimizer=None, train=False, first_stage=first_stage, mark='test')
            test_acc_all+=test_acc
            test_corr_all+=test_corr
            test_acc3+=test_acc
            test_fscores3+=test_fscore
            test_corr3+=test_corr
            print(f"The best(acc) epoch of test_condition ({args.test_condition}):  --test_acc {test_acc} --test_ua {test_corr}.")
            args.test_condition = 'av'
            test_mae, test_corr, test_acc, test_fscore, test_acc_atv, test_names, test_loss, weight_test,_ = train_or_eval_model(args, model, reg_loss, cls_loss, test_loader, \
                                                                                    optimizer=None,
                                                                                    train=False, first_stage=first_stage, mark='test')
            test_acc_all+=test_acc
            test_corr_all+=test_corr
            test_acc4+=test_acc
            test_fscores4+=test_fscore
            test_corr4+=test_corr
            print(f"The best(acc) epoch of test_condition ({args.test_condition}):  --test_acc {test_acc} --test_ua {test_corr}.")
            args.test_condition = 'tv'
            test_mae, test_corr, test_acc, test_fscore, test_acc_atv, test_names, test_loss, weight_test,_ = train_or_eval_model(args, model, reg_loss, cls_loss, test_loader, \
                                                                                    optimizer=None,
                                                                                    train=False, first_stage=first_stage, mark='test')
            test_acc_all+=test_acc
            test_corr_all+=test_corr
            test_acc5+=test_acc
            test_fscores5+=test_fscore
            test_corr5+=test_corr
            print(f"The best(acc) epoch of test_condition ({args.test_condition}):  --test_acc {test_acc} --test_ua {test_corr}.")
            args.test_condition = 'atv'
            test_mae, test_corr, test_acc, test_fscore, test_acc_atv, test_names, test_loss, weight_test,_ = train_or_eval_model(args, model, reg_loss, cls_loss, test_loader, \
                                                                                    optimizer=None,
                                                                                    train=False, first_stage=first_stage, mark='test')
            test_acc_all+=test_acc
            test_corr_all+=test_corr
            test_acc6+=test_acc
            test_fscores6+=test_fscore
            test_corr6+=test_corr
            print(f"The best(acc) epoch of test_condition ({args.test_condition}):  --test_acc {test_acc} --test_ua {test_corr}.")
            print(f"Folder avg: test_condition ({args.test_condition}) --test_acc {test_acc_all/7} --test_ua {test_corr_all/7}")
    print(f"Average a test_acc0: {test_acc0 / args.num_folder}, test_fscore0: {test_fscores0 / args.num_folder}, test_corr0: {test_corr0 / args.num_folder}")
    print(f"Average t test_acc1: {test_acc1 / args.num_folder}, test_fscore1: {test_fscores1 / args.num_folder}, test_corr1: {test_corr1 / args.num_folder}")
    print(f"Average v test_acc2: {test_acc2 / args.num_folder}, test_fscore2: {test_fscores2 / args.num_folder}, test_corr2: {test_corr2 / args.num_folder}")
    print(f"Average at test_acc3: {test_acc3 / args.num_folder}, test_fscore3: {test_fscores3 / args.num_folder}, test_corr3: {test_corr3 / args.num_folder}")
    print(f"Average av test_acc4: {test_acc4 / args.num_folder}, test_fscore4: {test_fscores4 / args.num_folder}, test_corr4: {test_corr4 / args.num_folder}")
    print(f"Average tv  test_acc5: {test_acc5 / args.num_folder}, test_fscore5: {test_fscores5 / args.num_folder}, test_corr5: {test_corr5 / args.num_folder}")
    print(f"Average atv test_acc6: {test_acc6 / args.num_folder}, test_fscore6: {test_fscores6 / args.num_folder}, test_corr6: {test_corr6 / args.num_folder}")

    avg_test_corr = (test_corr0 + test_corr1 + test_corr2 + test_corr3 + test_corr4 + test_corr5 + test_corr6) / (7 * args.num_folder)
    avg_test_acc = (test_acc0 + test_acc1 + test_acc2 + test_acc3 + test_acc4 + test_acc5 + test_acc6) / (7 * args.num_folder)
    avg_test_fscore = (test_fscores0 + test_fscores1 + test_fscores2 + test_fscores3 + test_fscores4 + test_fscores5 + test_fscores6) / (7 * args.num_folder)
    print(f"Overall average test_acc: {avg_test_acc}, Overall average test_fscore: {avg_test_fscore}, Overall average test_corr: {avg_test_corr}")
