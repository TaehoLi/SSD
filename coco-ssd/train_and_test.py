import torch
import os

num_training_loss = 0 #global variable

def train(loader, net, criterion, optimizer, device, checkpoint_folder, debug_steps=100, epoch=-1):
    global num_training_loss

    net.train(True)
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    
    for i, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        confidence, locations = net(images)
        
        regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)  # TODO CHANGE BOXES
        loss = regression_loss + classification_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
        
        if i and i % (debug_steps-1) == 0:
            num_training_loss += 1
            name = "vgg16-ssd" + "-Train-" + "%d"%num_training_loss + "-running_Loss-" + "%s"%str(running_loss)[:7] + ".pth"
            model_path = os.path.join(checkpoint_folder, name)
            #net.save(model_path)
            #torch.save(net.state_dict(), model_path)
            #print("Saved model:", model_path, "\n")
            
            running_loss = 0.0
            running_regression_loss = 0.0
            running_classification_loss = 0.0
            
    return

def test(loader, net, criterion, device):
    net.eval()
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    num = 0
    
    for _, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        num += 1

        with torch.no_grad():
            confidence, locations = net(images)
            regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
            loss = regression_loss + classification_loss

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
        
    return running_loss / num, running_regression_loss / num, running_classification_loss / num

