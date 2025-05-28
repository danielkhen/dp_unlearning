import static
import time


import math
import torch
import torch.nn as nn
from memory_bank import MemoryBank

def get_gradient(model:nn.Module):
    gradient = []
    for _, param in model.named_parameters():
        if param.requires_grad:
            grad = param.grad.clone().detach()
            gradient.append(grad.view(-1))

    return gradient    


def rectify_graident(grads_x, grads_y):
    r_grads_x = []
    r_grads_y = []

    for x, y in zip(grads_x, grads_y):
        if torch.cosine_similarity(x, y, dim=0) < 0:
            InP_xy = torch.matmul(y, x) 
            Inp_xx = torch.norm(x, p=2) ** 2
            Inp_yy = torch.norm(y, p=2) ** 2
            x = x - InP_xy/Inp_yy * y
            y = y - InP_xy/Inp_xx * x

        r_grads_x.append(x)
        r_grads_y.append(y)

    return r_grads_x, r_grads_y


def val_model(model:nn.Module, test_loader):
    criterion = nn.CrossEntropyLoss(reduction='sum')
    model.eval()
    val_loss = 0.0
    val_corrects = 0
    with torch.no_grad():
        for idx, (data, targets) in enumerate(test_loader):
            data, targets = data.to(static.CUDA), targets.to(static.CUDA)
            outputs = model(data)
            loss = criterion(outputs, targets)
            pred = torch.argmax(outputs, dim=1)
            
            val_loss += loss.item()
            val_corrects += torch.sum(pred == targets)

    val_loss /= len(test_loader.dataset)
    val_acc = val_corrects.double() / len(test_loader.dataset)
    
    return val_loss, val_acc
            
def unlearn_model(model:nn.Module, criterion, optimizer, data_loaders, num_epochs=200, batch_size=128, weights_path='weights/unlearned_model.pth'):
    t_dataset_sizes= len(data_loaders['unlearn'].dataset)
    bank = MemoryBank(size=math.ceil(t_dataset_sizes/batch_size))
    

    for epoch in range(num_epochs):
        begin_time = time.time()
        running_loss= 0.0
        running_corrects = 0.0
        
        for (t_data, t_labels), (n_data, n_labels)in zip(data_loaders['unlearn'], data_loaders['remain']):
            model.train()
            n_data, n_labels = n_data.to(static.CUDA), n_labels.to(static.CUDA)
            optimizer.zero_grad()
            n_outputs = model(n_data)
            n_loss = criterion(n_outputs, n_labels)
            n_loss.backward()
            
            n_grads = get_gradient(model)

            t_data, t_labels = t_data.to(static.CUDA), t_labels.to(static.CUDA)
            t_outputs = model(t_data)
            optimizer.zero_grad()

            t_loss = -criterion(t_outputs, t_labels)
            t_loss.backward()

            t_grads = get_gradient(model)
            
            
            bank.update(t_grads[-1])
            

            # l1_norm = sum([torch.norm(x, p=1) for x in model.parameters()])
            # optimizer.zero_grad()
            # l1_norm.backward()
            # l1_norm_grads = get_gradient(model)

            r_n_grads, r_t_grads = rectify_graident(n_grads, t_grads)
            if epoch > 0 and bank.mean_grads(r_t_grads[-1]) != None:
                grads, _ = rectify_graident([r_t_grads[-1]], [bank.mean_grads(r_t_grads[-1])])
                r_t_grads[-1] = grads[-1]
                
            with torch.no_grad():
                    gamma, epsilon = 100, 0.02
                    lambda_weight = 1/(1+torch.exp(gamma*(n_loss-epsilon)))

            optimizer.zero_grad()
            idx = 0
            for _, param in model.named_parameters():
                if param.requires_grad:
                    param.grad =  ((1-lambda_weight)*r_n_grads[idx]+lambda_weight*r_t_grads[idx]).view(param.size())
                    idx += 1

            optimizer.step()

            
            preds = torch.argmax(t_outputs.data, dim=1)
            running_loss += t_loss.item()
            running_corrects += torch.sum(preds == t_labels.data)

        
        #schedule.step()
        end_time = time.time()
        epoch_loss = running_loss / len(data_loaders['unlearn'])
        epoch_acc = float(running_corrects) / len(data_loaders['unlearn'].dataset)

        u_loss, u_acc = val_model(model, data_loaders['unlearn'])
        r_loss, r_acc = val_model(model, data_loaders['remain'])
        v_loss, v_acc = val_model(model, data_loaders['val'])

        print(f'Epoch: {epoch} - u_loss: {u_loss:.4f} - u_acc: {u_acc:.4f} - r_loss: {r_loss:.4f} - r_acc: {r_acc:.4f} - v_loss: {v_loss:.4f} - val_acc: {v_acc:.4f} - time: {end_time - begin_time:.2f}s')

        if v_acc > u_acc:
            print("Forget accuracy is lower than test accuracy")
            break
        
    # Save model weightsmodel
    torch.save({'model': model.state_dict()}, weights_path)

        # if args.save_checkpoint:
        #     checkpoint_dir = os.path.join(args.exp_dir, args.dataset + '-' + args.model + '-' + str(args.unlearned_size))

        #     os.makedirs(checkpoint_dir, exist_ok=True)
                
        #     checkpoint_path = os.path.join(checkpoint_dir, 'epoch_%d.pt' % epoch)
            
    
        #     torch.save({
        #             'epoch': epoch,
        #             'model_state_dict': model.state_dict(),
        #             'optimizer_state_dict': optimizer.state_dict(),
        #             'u_loss': u_loss,
        #             'u_acc': u_acc,
        #             'r_loss':r_loss,
        #             'r_acc': r_acc,
        #             'v_loss': v_loss,
        #             'v_acc': v_acc,
        #         }, checkpoint_path)

        
    