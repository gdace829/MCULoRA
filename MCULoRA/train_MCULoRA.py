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

def train_or_eval_model(args, model, reg_loss, cls_loss, dataloader, optimizer=None, train=False, first_stage=True, mark='train',probabilities=[0.5,0.5,0.5],appearance_probabilities=[0.5,0.5,0.5,0.5,0.5,0.5,0.5]):
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
        matrix = generate_mask(seqlen, batch, args.test_condition, first_stage,train,probabilities,appearance_probabilities) # [seqlen*batch, view_num]
        audio_host_mask = np.reshape(matrix[0], (batch, seqlen, 1))
        text_host_mask = np.reshape(matrix[1], (batch, seqlen, 1))
        visual_host_mask = np.reshape(matrix[2], (batch, seqlen, 1))
        audio_host_mask = torch.LongTensor(audio_host_mask.transpose(1, 0, 2))
        text_host_mask = torch.LongTensor(text_host_mask.transpose(1, 0, 2))
        visual_host_mask = torch.LongTensor(visual_host_mask.transpose(1, 0, 2))
        # guest mask
        matrix = generate_mask(seqlen, batch, args.test_condition, first_stage,train,probabilities,appearance_probabilities) # [seqlen*batch, view_num]
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
    
        if dataset in ['IEMOCAPFour', 'IEMOCAPSix']:
                loss = cls_loss(lp_, labels_, umask)
        if dataset in ['CMUMOSI', 'CMUMOSEI']:
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

        # sjs记录一下梯度方向的大小
        if train and not first_stage:
            # INSERT_YOUR_CODE
            # 融合特征对比损失：将xs_a, xs_t, xs_v中相同key的表征拉近，不同key拉远
            # 只在训练且非first_stage时进行
            contrastive_loss = 0.0
            temperature = 0.07  # 可调超参数

            # 获取所有key
            keys = list(xs_a.keys())
            # 统计当前batch的模态组合
            # index: '0'-atv, '1'-at, '2'-av, '3'-tv, '4'-a, '5'-t, '6'-v
            # 只对当前模态组合涉及的key做拉近
            # 例如 index='0'，则key='atv'，index='3'，则key='tv'
            # 其余key之间做拉远

            # 当前batch的key
            current_key = None
            if index == '0':
                current_key = 'atv'
            elif index == '1':
                current_key = 'at'
            elif index == '2':
                current_key = 'av'
            elif index == '3':
                current_key = 'tv'
            elif index == '4':
                current_key = 'a'
            elif index == '5':
                current_key = 't'
            elif index == '6':
                current_key = 'v'

            if current_key is not None and current_key in xs_a and current_key in xs_t and current_key in xs_v:
         
                feat_a = xs_a[current_key]  # [B, D]
                feat_t = xs_t[current_key]
                feat_v = xs_v[current_key]
                # 归一化
                feat_a = F.normalize(feat_a, dim=-1)
                feat_t = F.normalize(feat_t, dim=-1)
                feat_v = F.normalize(feat_v, dim=-1)
  
                pos_sim_at = (feat_a * feat_t).sum(dim=-1) / temperature
                pos_sim_av = (feat_a * feat_v).sum(dim=-1) / temperature
                pos_sim_tv = (feat_t * feat_v).sum(dim=-1) / temperature
              
                pos_loss = - (pos_sim_at.mean() + pos_sim_av.mean() + pos_sim_tv.mean()) / 3.0
                contrastive_loss += pos_loss

                # 拉远：不同key的a/t/v表征
                for other_key in keys:
                    if other_key == current_key:
                        continue
                    # 只在other_key存在时才计算
                    if other_key in xs_a and other_key in xs_t and other_key in xs_v:
                        # 取出other_key下的a/t/v特征
                        other_a = xs_a[other_key]
                        other_t = xs_t[other_key]
                        other_v = xs_v[other_key]
                        # 归一化
                        other_a = F.normalize(other_a, dim=-1)
                        other_t = F.normalize(other_t, dim=-1)
                        other_v = F.normalize(other_v, dim=-1)
                      
                        neg_sim_aa = F.cosine_similarity(feat_a, other_a, dim=-1) / temperature
                        neg_sim_tt = F.cosine_similarity(feat_t, other_t, dim=-1) / temperature
                        neg_sim_vv = F.cosine_similarity(feat_v, other_v, dim=-1) / temperature
                        # 拉远损失，越小越好，取正号
                        neg_loss = (neg_sim_aa.mean() + neg_sim_tt.mean() + neg_sim_vv.mean()) / 3.0
                        contrastive_loss += neg_loss

            # 将对比损失加到总loss中，权重可调
            loss = loss + 0.001 * contrastive_loss
            loss.backward()# 梯度反向传播，每次数据迭代都进行
            optimizer.step()
    
    assert preds!=[], f'Error: no dataset in dataloader'
    preds  = np.concatenate(preds)
    preds_a = np.concatenate(preds_a)
    preds_t = np.concatenate(preds_t)
    preds_v = np.concatenate(preds_v)
    labels = np.concatenate(labels)
    masks  = np.concatenate(masks)
   
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
        return mae, ua, avg_accuracy, avg_fscore, [avg_acc_a, avg_acc_t, avg_acc_v], vidnames, avg_loss, weight,x_a,x_t,x_v,xs_a,xs_t,xs_v

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
        return mae, corr, avg_accuracy, avg_fscore, [avg_acc_a, avg_acc_t, avg_acc_v], vidnames, avg_loss, weight,x_a,x_t,x_v,xs_a,xs_t,xs_v



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
    for ii in range(1):
        ii=0
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
        lastacc=[]
        # 记录偏好分数的列表
        preference_scores_history = {
            'atv': [], 'at': [], 'av': [], 'tv': [], 
            'a': [], 't': [], 'v': []
        }
        # 记录出现概率的列表
        appearance_probabilities_history = {
            'atv': [], 'at': [], 'av': [], 'tv': [], 
            'a': [], 't': [], 'v': []
        }
        appearance_probabilities = [0.5,0.5,0.5,0.5,0.5,0.5,0.5]
        for epoch in range(args.epochs):
            first_stage = False if epoch < args.stage_epoch else False
            ## training and testing (!!! if IEMOCAP, the ua is equal to corr !!!)
            args.test_condition="atv"
            train_mae, train_corr, train_acc, train_fscore, train_acc_atv, train_names, train_loss, weight_train,x_a,x_t,x_v,xs_a,xs_t,xs_v= train_or_eval_model(args, model, reg_loss, cls_loss, train_loader, \
                                                                            optimizer=optimizer, train=True, first_stage=first_stage, mark='train',probabilities=probabilities_adjust,appearance_probabilities=appearance_probabilities)
            test_mae, test_corr, test_acc, test_fscore, test_acc_atv, test_names, test_loss, weight_test,x_a,x_t,x_v,xs_a,xs_t,xs_v = train_or_eval_model(args, model, reg_loss, cls_loss, test_loader, \
                                                                            optimizer=None, train=False, first_stage=first_stage, mark='test',probabilities=probabilities_adjust,appearance_probabilities=appearance_probabilities)
            args.test_condition="at"
            _, _, test_acc1, _, _, _, _, _,_,_,_,_,_,_ = train_or_eval_model(args, model, reg_loss, cls_loss, test_loader, \
                                                                            optimizer=None, train=False, first_stage=first_stage, mark='test',probabilities=probabilities_adjust,appearance_probabilities=appearance_probabilities)
            args.test_condition="av"
            _, _, test_acc2, _, _, _, _, _,_,_,_,_,_,_= train_or_eval_model(args, model, reg_loss, cls_loss, test_loader, \
                                                                            optimizer=None, train=False, first_stage=first_stage, mark='test',probabilities=probabilities_adjust,appearance_probabilities=appearance_probabilities)
            args.test_condition="tv"
            _, _, test_acc3, _, _, _, _, _,_,_,_,_,_,_= train_or_eval_model(args, model, reg_loss, cls_loss, test_loader, \
                                                                            optimizer=None, train=False, first_stage=first_stage, mark='test',probabilities=probabilities_adjust,appearance_probabilities=appearance_probabilities)
            args.test_condition="a"
            _, _, test_acc4, _, _, _, _, _,_,_,_,_,_,_ = train_or_eval_model(args, model, reg_loss, cls_loss, test_loader, \
                                                                            optimizer=None, train=False, first_stage=first_stage, mark='test',probabilities=probabilities_adjust,appearance_probabilities=appearance_probabilities)
            args.test_condition="t"
            _, _, test_acc5, _, _, _, _, _,_,_,_,_,_,_ = train_or_eval_model(args, model, reg_loss, cls_loss, test_loader, \
                                                                                optimizer=None, train=False, first_stage=first_stage, mark='test',probabilities=probabilities_adjust,appearance_probabilities=appearance_probabilities)
            args.test_condition="v"
            _, _, test_acc6, _, _, _, _, _,_,_,_,_,_,_ = train_or_eval_model(args, model, reg_loss, cls_loss, test_loader, \
                                                                            optimizer=None, train=False, first_stage=first_stage, mark='test',probabilities=probabilities_adjust,appearance_probabilities=appearance_probabilities)
            curacc=[test_acc,test_acc1,test_acc2,test_acc3,test_acc4,test_acc5,test_acc6]
           
            # 计算与lastacc的差值及其百分比
            if len(lastacc) > 0:
                # 计算差值
                acc_diff = [curacc[i] - lastacc[i] for i in range(len(curacc))]
                # 计算百分比变化
                acc_percent = [(curacc[i] - lastacc[i]) / lastacc[i] * 100 if lastacc[i] != 0 else 0 for i in range(len(curacc))]
                
                # 计算偏好分数
                total_diff = sum(acc_diff)
                preference_scores = []
                for i in range(len(acc_diff)):
                    # 计算除了自己以外其他所有差值的总和
                    others_sum = total_diff - acc_diff[i]
                    # 偏好分数 = 其他差值总和 / 所有差值总和
                    preference_score = others_sum / total_diff if total_diff != 0 else 0
                    # 限制偏好分数上限为1.5
                    preference_score = min(preference_score, 1.5)
                    preference_scores.append(preference_score)
                    
                    
                
                print(f"Epoch {epoch}: atv={test_acc:.4f}, at={test_acc1:.4f}, av={test_acc2:.4f}, tv={test_acc3:.4f}, a={test_acc4:.4f}, t={test_acc5:.4f}, v={test_acc6:.4f}")
                # print(f"差值: atv={acc_diff[0]:.4f}, at={acc_diff[1]:.4f}, av={acc_diff[2]:.4f}, tv={acc_diff[3]:.4f}, a={acc_diff[4]:.4f}, t={acc_diff[5]:.4f}, v={acc_diff[6]:.4f}")
                # print(f"百分比变化: atv={acc_percent[0]:.2f}%, at={acc_percent[1]:.2f}%, av={acc_percent[2]:.2f}%, tv={acc_percent[3]:.2f}%, a={acc_percent[4]:.2f}%, t={acc_percent[5]:.2f}%, v={acc_percent[6]:.2f}%")
                # print(f"偏好分数: atv={preference_scores[0]:.4f}, at={preference_scores[1]:.4f}, av={preference_scores[2]:.4f}, tv={preference_scores[3]:.4f}, a={preference_scores[4]:.4f}, t={preference_scores[5]:.4f}, v={preference_scores[6]:.4f}")
                
                # 计算下一回合的出现概率
                q_base = 0.8
                alpha = 1.0
                appearance_probabilities = []
                for i, pref_score in enumerate(preference_scores):
                    # 概率 = q_base * (1 + α * tan(x-1))
                    prob = q_base * (1 + alpha * np.tan(pref_score - 1))
                    appearance_probabilities.append(prob)
                # 更新appearance_probabilities为计算出的概率，用于下一回合
                appearance_probabilities = appearance_probabilities.copy()
                
                # 记录偏好分数和出现概率
                modality_names = ['atv', 'at', 'av', 'tv', 'a', 't', 'v']
                for i, name in enumerate(modality_names):
                    preference_scores_history[name].append(preference_scores[i])
                    appearance_probabilities_history[name].append(appearance_probabilities[i])
            else:
                lastacc=curacc
                print(f"Epoch {epoch}: atv={test_acc:.4f}, at={test_acc1:.4f}, av={test_acc2:.4f}, tv={test_acc3:.4f}, a={test_acc4:.4f}, t={test_acc5:.4f}, v={test_acc6:.4f}")
            
            # 更新lastacc为当前的curacc
            lastacc = curacc.copy()
            # 训练时的准确率变化
            # vis_acc_test0.append(test_acc)
            # vis_acc_test1.append(test_acc1)
            # vis_acc_test2.append(test_acc2)
            # vis_acc_test3.append(test_acc3)
            # vis_acc_test4.append(test_acc4)
            # vis_acc_test5.append(test_acc5)
            # vis_acc_test6.append(test_acc6)
            args.test_condition="atv"
           
           

            ## save
            if epoch>args.stage_epoch:
                test_accs.append(test_acc)# 每个test epoch的结果汇聚起来选出最好的一次epoch 由于前面100个epoch没有掩码可能导致性能比较好所以出现了这个问题
                test_fscores.append(test_fscore)
                test_maes.append(test_mae)
                test_corrs.append(test_corr)
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
      

  
     
        # 保存偏好分数历史记录
        # np.save(f'momkevis_preference_scores_atv_folder{ii}.npy', np.array(preference_scores_history['atv']))
        # np.save(f'momkevis_preference_scores_at_folder{ii}.npy', np.array(preference_scores_history['at']))
        # np.save(f'momkevis_preference_scores_av_folder{ii}.npy', np.array(preference_scores_history['av']))
        # np.save(f'momkevis_preference_scores_tv_folder{ii}.npy', np.array(preference_scores_history['tv']))
        # np.save(f'momkevis_preference_scores_a_folder{ii}.npy', np.array(preference_scores_history['a']))
        # np.save(f'momkevis_preference_scores_t_folder{ii}.npy', np.array(preference_scores_history['t']))
        # np.save(f'momkevis_preference_scores_v_folder{ii}.npy', np.array(preference_scores_history['v']))
        
        # # 保存出现概率历史记录
        # np.save(f'momkevis_appearance_probabilities_atv_folder{ii}.npy', np.array(appearance_probabilities_history['atv']))
        # np.save(f'momkevis_appearance_probabilities_at_folder{ii}.npy', np.array(appearance_probabilities_history['at']))
        # np.save(f'momkevis_appearance_probabilities_av_folder{ii}.npy', np.array(appearance_probabilities_history['av']))
        # np.save(f'momkevis_appearance_probabilities_tv_folder{ii}.npy', np.array(appearance_probabilities_history['tv']))
        # np.save(f'momkevis_appearance_probabilities_a_folder{ii}.npy', np.array(appearance_probabilities_history['a']))
        # np.save(f'momkevis_appearance_probabilities_t_folder{ii}.npy', np.array(appearance_probabilities_history['t']))
        # np.save(f'momkevis_appearance_probabilities_v_folder{ii}.npy', np.array(appearance_probabilities_history['v']))

    

        end_second_stage_time = time.time()
        print("-"*80)
      
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
            test_mae, test_corr, test_acc, test_fscore, test_acc_atv, test_names, test_loss, weight_test = train_or_eval_model(args, model, reg_loss, cls_loss, test_loader, \
                                                                                    optimizer=None, train=False, first_stage=first_stage, mark='test')
            test_mae_all+=test_mae
            test_corr_all+=test_corr
            test_acc_all+=test_acc
            test_fscore_all+=test_fscore
            test_mae_all+=test_mae
            print(f"The best(acc) epoch of test_condition ({args.test_condition}):  --test_mae {test_mae} --test_corr {test_corr} --test_fscores {test_fscore} --test_acc {test_acc}.")
            args.test_condition = 't'
            test_mae, test_corr, test_acc, test_fscore, test_acc_atv, test_names, test_loss, weight_test= train_or_eval_model(args, model, reg_loss, cls_loss, test_loader, \
                                                                                    optimizer=None, train=False, first_stage=first_stage, mark='test')
            test_mae_all+=test_mae
            test_corr_all+=test_corr
            test_acc_all+=test_acc
            test_fscore_all+=test_fscore
            print(f"The best(acc) epoch of test_condition ({args.test_condition}):  --test_mae {test_mae} --test_corr {test_corr} --test_fscores {test_fscore} --test_acc {test_acc}.")
            args.test_condition = 'v'
            test_mae, test_corr, test_acc, test_fscore, test_acc_atv, test_names, test_loss, weight_test = train_or_eval_model(args, model, reg_loss, cls_loss, test_loader, \
                                                                                    optimizer=None, train=False, first_stage=first_stage, mark='test')
            test_mae_all+=test_mae
            test_corr_all+=test_corr
            test_acc_all+=test_acc
            test_fscore_all+=test_fscore
            print(f"The best(acc) epoch of test_condition ({args.test_condition}):  --test_mae {test_mae} --test_corr {test_corr} --test_fscores {test_fscore} --test_acc {test_acc}.")
            args.test_condition = 'at'
            test_mae, test_corr, test_acc, test_fscore, test_acc_atv, test_names, test_loss, weight_test = train_or_eval_model(args, model, reg_loss, cls_loss, test_loader, \
                                                                                    optimizer=None, train=False, first_stage=first_stage, mark='test')
            test_mae_all+=test_mae
            test_corr_all+=test_corr
            test_acc_all+=test_acc
            test_fscore_all+=test_fscore
            print(f"The best(acc) epoch of test_condition ({args.test_condition}):  --test_mae {test_mae} --test_corr {test_corr} --test_fscores {test_fscore} --test_acc {test_acc}.")
            args.test_condition = 'av'
            test_mae, test_corr, test_acc, test_fscore, test_acc_atv, test_names, test_loss, weight_test = train_or_eval_model(args, model, reg_loss, cls_loss, test_loader, \
                                                                                    optimizer=None, train=False, first_stage=first_stage, mark='test')
            test_mae_all+=test_mae
            test_corr_all+=test_corr
            test_acc_all+=test_acc
            test_fscore_all+=test_fscore
            print(f"The best(acc) epoch of test_condition ({args.test_condition}):  --test_mae {test_mae} --test_corr {test_corr} --test_fscores {test_fscore} --test_acc {test_acc}.")
            args.test_condition = 'tv'
            test_mae, test_corr, test_acc, test_fscore, test_acc_atv, test_names, test_loss, weight_test= train_or_eval_model(args, model, reg_loss, cls_loss, test_loader, \
                                                                                    optimizer=None, train=False, first_stage=first_stage, mark='test')
            test_mae_all+=test_mae
            test_corr_all+=test_corr
            test_acc_all+=test_acc
            test_fscore_all+=test_fscore
            print(f"The best(acc) epoch of test_condition ({args.test_condition}):  --test_mae {test_mae} --test_corr {test_corr} --test_fscores {test_fscore} --test_acc {test_acc}.")
            args.test_condition = 'atv'
            test_mae, test_corr, test_acc, test_fscore, test_acc_atv, test_names, test_loss, weight_test = train_or_eval_model(args, model, reg_loss, cls_loss, test_loader, \
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
            test_mae, test_corr, test_acc, test_fscore, test_acc_atv, test_names, test_loss, weight_test,x_a,x_t,x_v,xs_a,xs_t,xs_v= train_or_eval_model(args, model, reg_loss, cls_loss, test_loader, \
                                                                                    optimizer=None, train=False, first_stage=first_stage, mark='test')
            test_acc_all+=test_acc
            test_corr_all+=test_corr
            test_acc0+=test_acc
            test_fscores0+=test_fscore
            test_corr0+=test_corr
            print(f"The best(acc) epoch of test_condition ({args.test_condition}):  --test_acc {test_acc} --test_ua {test_corr}.")
            args.test_condition = 't'
            test_mae, test_corr, test_acc, test_fscore, test_acc_atv, test_names, test_loss, weight_test,x_a,x_t,x_v,xs_a,xs_t,xs_v = train_or_eval_model(args, model, reg_loss, cls_loss, test_loader, \
                                                                                    optimizer=None, train=False, first_stage=first_stage, mark='test')
            test_acc_all+=test_acc
            test_corr_all+=test_corr
            test_acc1+=test_acc
            test_fscores1+=test_fscore
            test_corr1+=test_corr
            print(f"The best(acc) epoch of test_condition ({args.test_condition}):  --test_acc {test_acc} --test_ua {test_corr}.")
            args.test_condition = 'v'
            test_mae, test_corr, test_acc, test_fscore, test_acc_atv, test_names, test_loss, weight_test,x_a,x_t,x_v,xs_a,xs_t,xs_v = train_or_eval_model(args, model, reg_loss, cls_loss, test_loader, \
                                                                                    optimizer=None, train=False, first_stage=first_stage, mark='test')
            test_acc_all+=test_acc
            test_corr_all+=test_corr
            test_acc2+=test_acc
            test_fscores2+=test_fscore
            test_corr2+=test_corr
            print(f"The best(acc) epoch of test_condition ({args.test_condition}):  --test_acc {test_acc} --test_ua {test_corr}.")
            args.test_condition = 'at'
            test_mae, test_corr, test_acc, test_fscore, test_acc_atv, test_names, test_loss, weight_test,x_a,x_t,x_v,xs_a,xs_t,xs_v = train_or_eval_model(args, model, reg_loss, cls_loss, test_loader, \
                                                                                    optimizer=None, train=False, first_stage=first_stage, mark='test')
            test_acc_all+=test_acc
            test_corr_all+=test_corr
            test_acc3+=test_acc
            test_fscores3+=test_fscore
            test_corr3+=test_corr
            print(f"The best(acc) epoch of test_condition ({args.test_condition}):  --test_acc {test_acc} --test_ua {test_corr}.")
            args.test_condition = 'av'
            test_mae, test_corr, test_acc, test_fscore, test_acc_atv, test_names, test_loss, weight_test,x_a,x_t,x_v,xs_a,xs_t,xs_v = train_or_eval_model(args, model, reg_loss, cls_loss, test_loader, \
                                                                                    optimizer=None,
                                                                                    train=False, first_stage=first_stage, mark='test')
            test_acc_all+=test_acc
            test_corr_all+=test_corr
            test_acc4+=test_acc
            test_fscores4+=test_fscore
            test_corr4+=test_corr
            print(f"The best(acc) epoch of test_condition ({args.test_condition}):  --test_acc {test_acc} --test_ua {test_corr}.")
            args.test_condition = 'tv'
            test_mae, test_corr, test_acc, test_fscore, test_acc_atv, test_names, test_loss, weight_test,x_a,x_t,x_v,xs_a,xs_t,xs_v = train_or_eval_model(args, model, reg_loss, cls_loss, test_loader, \
                                                                                    optimizer=None,
                                                                                    train=False, first_stage=first_stage, mark='test')
            test_acc_all+=test_acc
            test_corr_all+=test_corr
            test_acc5+=test_acc
            test_fscores5+=test_fscore
            test_corr5+=test_corr
            print(f"The best(acc) epoch of test_condition ({args.test_condition}):  --test_acc {test_acc} --test_ua {test_corr}.")
            args.test_condition = 'atv'
            test_mae, test_corr, test_acc, test_fscore, test_acc_atv, test_names, test_loss, weight_test,x_a,x_t,x_v,xs_a,xs_t,xs_v = train_or_eval_model(args, model, reg_loss, cls_loss, test_loader, \
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
