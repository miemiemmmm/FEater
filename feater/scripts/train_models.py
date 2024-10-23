import os, sys, time, io
import argparse, random, json 

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import transformers

from torch.utils.tensorboard import SummaryWriter

# Import models 
from feater.models.pointnet import PointNetCls      
from feater import dataloader, utils
import feater

tensorboard_writer = None 
args = None

# For point cloud type of data, the input is in the shape of (B, 3, N)
INPUT_POINTS = 0
def update_pointnr(pointnr):
  global INPUT_POINTS
  INPUT_POINTS = pointnr
DATALOADER_TYPE = ""
def update_dataloader_type(dataloader_type):
  global DATALOADER_TYPE
  DATALOADER_TYPE = dataloader_type

OPTIMIZERS = {
  "adam": optim.Adam, 
  "sgd": optim.SGD,
  "adamw": optim.AdamW,
}

LOSS_FUNCTIONS = {
  "crossentropy": nn.CrossEntropyLoss,
}

DATALOADER_TYPES = {
  "pointnet": dataloader.CoordDataset,
  "pointnet2": dataloader.CoordDataset,
  "dgcnn": dataloader.CoordDataset,
  "paconv": dataloader.CoordDataset,

  
  "voxnet": dataloader.VoxelDataset,
  "deeprank": dataloader.VoxelDataset,
  "gnina": dataloader.VoxelDataset,

  
  "resnet": dataloader.HilbertCurveDataset,
  "convnext": dataloader.HilbertCurveDataset,
  "convnext_iso": dataloader.HilbertCurveDataset,
  "swintrans": dataloader.HilbertCurveDataset,
  "ViT": dataloader.HilbertCurveDataset,

  "coord": dataloader.CoordDataset, 
  "surface": dataloader.SurfDataset, 
}




def get_model(model_type:str, output_dim:int): 
  if model_type == "pointnet":
    model = PointNetCls(output_dim)

  elif model_type == "pointnet2":
    from feater.models.pointnet2 import get_model as get_pointnet2_model
    if DATALOADER_TYPE == "surface": 
      rads = [0.5, 1.00]  # For surface-based training
    elif DATALOADER_TYPE == "coord":
      rads = [1.75, 3.60]  # For coordinate-based data
    else: 
      raise ValueError(f"Unexpected dataloader type {DATALOADER_TYPE} for PointNet2 model; Only surface and coord are supported")
    print(f"Using the radii {rads} for the PointNet2 model")
    model = get_pointnet2_model(output_dim, normal_channel=False, sample_nr = INPUT_POINTS, rads=rads)
    
  elif model_type == "dgcnn":
    from feater.models.dgcnn import DGCNN_cls
    args = {
      "k": 20, 
      "emb_dims": 1024,
      "dropout" : 0.25,
    }
    model = DGCNN_cls(args, output_channels=output_dim)
  elif model_type == "paconv":
    # NOTE: This is a special case due to the special dependency of the PAConv !!!!
    if "/MieT5/tests/PAConv/obj_cls" not in sys.path:
      sys.path.append("/MieT5/tests/PAConv/obj_cls")
    from feater.models.paconv import PAConv
    config = {
      "k_neighbors": 20, 
      "output_channels": output_dim,
      "dropout": 0.25,
    }
    model = PAConv(config)
  elif model_type == "voxnet":
    from feater.models.voxnet import VoxNet
    model = VoxNet(output_dim)
  elif model_type == "deeprank":
    from feater.models.deeprank import DeepRankNetwork
    model = DeepRankNetwork(1, output_dim, 32)
  elif model_type == "gnina":
    from feater.models.gnina import GninaNetwork
    model = GninaNetwork(output_dim)
  elif model_type == "resnet":
    from feater.models.resnet import ResNet
    model = ResNet(1, output_dim, "resnet18")
  elif model_type == "convnext":
    from feater.models.convnext import ConvNeXt
    """
      in_chans=3, num_classes=1000, 
      depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
      drop_path_rate=0.,  layer_scale_init_value=1e-6, 
      head_init_scale=1.,
    """
    model = ConvNeXt(1, output_dim)

  elif model_type == "convnext_iso":
    from feater.models.convnext import ConvNeXt, ConvNeXtIsotropic
    model = ConvNeXtIsotropic(1, output_dim)

  elif model_type == "swintrans":
    from transformers import SwinForImageClassification, SwinConfig
    configuration = SwinConfig(
      image_size = 128, 
      num_channels = 1,
      num_labels = output_dim,
      window_size=4, 
    )
    model = SwinForImageClassification(configuration)

  elif model_type == "ViT":
    from transformers import ViTConfig, ViTForImageClassification
    configuration = ViTConfig(
      image_size = 128, 
      num_channels = 1, 
      num_labels = output_dim, 
      window_size=4, 
    )
    model = ViTForImageClassification(configuration)

  else:
    raise ValueError(f"Unexpected model type {model_type}; Only voxnet, pointnet, resnet, and deeprank are supported")
  return model


def match_data(pred, label):  
  predicted_labels = torch.argmax(pred, dim=1)
  plt.scatter(predicted_labels.cpu().detach().numpy(), label.cpu().detach().numpy(), s=4, c = np.arange(len(label))/len(label), cmap="inferno")
  plt.xlabel("Predicted")
  plt.ylabel("Actual")
  plt.title("Predicted vs. Actual")
  buf = io.BytesIO()
  plt.savefig(buf, format="png")
  buf.seek(0)
  Image.open(buf)
  image_tensor = torchvision.transforms.ToTensor()(Image.open(buf))
  plt.clf()
  buf.close()
  return image_tensor

def get_lr(optimizer):
  for param_group in optimizer.param_groups:
    return param_group['lr']


def test_model(model, dataset, criterion, test_number, batch_size, use_cuda=1, process_nr=32):
  test_loss = 0.0
  correct = 0
  c = 0
  c_samples = 0
  with torch.no_grad():
    model.eval()
    for data, target in dataset.mini_batches(batch_size=batch_size, process_nr=process_nr):
      # Correct way to handle the input data
      # For the PointNet, the data is in the shape of (B, 3, N)
      # Important: Swap the axis to make the coordinate as 3 input channels of the data
      # print(target.unique(return_counts=True))

      if isinstance(dataset, dataloader.CoordDataset) or isinstance(dataset, dataloader.SurfDataset):
        data = data.transpose(2, 1)  
      if use_cuda:
        data, target = data.cuda(), target.cuda()

      pred = model(data)

      if isinstance(model, PointNetCls) or isinstance(pred, tuple):
        pred = pred[0]
      
      # Get the logit if the huggingface models is used
      if isinstance(pred, transformers.file_utils.ModelOutput): 
        pred = pred.logits
      
      pred_choice = torch.argmax(pred, dim=1)
      correct += pred_choice.eq(target.data).cpu().sum()
      test_loss += criterion(pred, target).item()

      # Increament the counter for test sample count
      c_samples += len(data)
      c += 1
      if c_samples >= test_number:
        break
    test_loss /= c
    accuracy = correct / c_samples
    return test_loss, accuracy


def parse_args():
  parser = argparse.ArgumentParser(description="Train the models used in the FEater paper. ")
  # Model selection
  parser.add_argument("-m", "--model", type=str, required=True, help="The model to train")
  parser.add_argument("--optimizer", type=str, default="adam", help="The optimizer to use")
  parser.add_argument("--loss-function", type=str, default="crossentropy", help="The loss function to use")
  

  # Data files 
  parser.add_argument("-train", "--training-data", type=str, required=True, help="The file writes all of the absolute path of h5 files of training data set")
  parser.add_argument("-test", "--test-data", type=str, required=True, help="The file writes all of the absolute path of h5 files of test data set")
  parser.add_argument("-o", "--output_folder", type=str, required=True, help="The output folder to store the model and performance data")
  parser.add_argument("-w", "--data_workers", type=int, default=12, help="Number of workers for data loading")
  parser.add_argument("--test-number", type=int, default=4000, help="Number of test samples to use")
  parser.add_argument("--dataloader-type", type=str, default=None, help="Dataset to use")


  # Pretrained model and break point restart
  parser.add_argument("--pretrained", type=str, default=None, help="Pretrained model path")
  parser.add_argument("--start-epoch", type=int, default=0, help="Start epoch")


  # Training parameters
  parser.add_argument("-b", "--batch-size", type=int, default=128, help="Batch size")
  parser.add_argument("-e", "--epochs", type=int, default=1, help="Number of epochs")
  parser.add_argument("-lr", "--lr-init", type=float, default=0.001, help="Learning rate")
  parser.add_argument("--lr-decay-steps", type=int, default=30, help="Decay the learning rate every n steps")
  parser.add_argument("--lr-decay-rate", type=float, default=0.5, help="Decay the learning rate by this rate")
  

  # Miscellanenous
  parser.add_argument("-v", "--verbose", default=0, type=int, help="Verbose printing while training")
  parser.add_argument("-s", "--manualSeed", type=int, help="Manually set seed")
  parser.add_argument("--cuda", type=int, default=int(torch.cuda.is_available()), help="Use CUDA")
  parser.add_argument("--data-type", type=str, default="single", help="Dataset to use")
  parser.add_argument("--production", type=int, default=0, help="Production mode")


  # Special parameters for the point-cloud-like representations
  parser.add_argument("--pointnet-points", type=int, default=0, help="Num of points to use")

  args = parser.parse_args()  
  if not os.path.exists(args.training_data): 
    raise ValueError(f"The training data file {args.training_data} does not exist.")
  if not os.path.exists(args.test_data):
    raise ValueError(f"The test data file {args.test_data} does not exist.")
  
  if args.dataloader_type is None: 
    args.dataloader_type = args.model

  update_pointnr(args.pointnet_points)
  update_dataloader_type(args.dataloader_type)

  if not args.manualSeed:
    args.seed = int(time.perf_counter().__str__().split(".")[-1])
  else:
    args.seed = int(args.manualSeed)

  f"{args.dataloader_type}_{args.data_type}"

  if os.path.exists(args.output_folder):
    args.output_folder = os.path.abspath(args.output_folder)
  else: 
    os.makedirs(args.output_folder)
    args.output_folder = os.path.abspath(args.output_folder)
    print("Warning: The output folder does not exist. Created the folder {args.output_folder}.")

  # Change the type of the cuda flag
  if args.data_type == "single":
    args.class_nr = 20
  elif args.data_type == "dual":
    args.class_nr = 400
  elif args.data_type == "modelnet":
    args.class_nr = 40
  else:
    raise ValueError(f"Unexpected data type {args.data_type}; Only single (FEater-Single) and dual (FEater-Dual) are supported")
  
  return args


def perform_training(training_settings: dict): 
  USECUDA = training_settings["cuda"]
  MODEL_TYPE = training_settings["model"]

  START_EPOCH = training_settings["start_epoch"]
  EPOCH_NR = training_settings["epochs"]
  BATCH_SIZE = training_settings["batch_size"]
  WORKER_NR = training_settings["data_workers"]


  random.seed(training_settings["seed"])
  torch.manual_seed(training_settings["seed"])
  np.random.seed(training_settings["seed"])

  # Load the datasets
  trainingfiles = utils.checkfiles(training_settings["training_data"])
  testfiles = utils.checkfiles(training_settings["test_data"])
  if training_settings["dataloader_type"] in ("surface", "coord"): 
    if training_settings["data_type"] == "modelnet": 
      training_data = DATALOADER_TYPES[training_settings["dataloader_type"]](trainingfiles, target_np=training_settings["pointnet_points"], scale=True)
      test_data = DATALOADER_TYPES[training_settings["dataloader_type"]](testfiles, target_np=training_settings["pointnet_points"], scale=True)
    else: 
      training_data = DATALOADER_TYPES[training_settings["dataloader_type"]](trainingfiles, target_np=training_settings["pointnet_points"])
      test_data = DATALOADER_TYPES[training_settings["dataloader_type"]](testfiles, target_np=training_settings["pointnet_points"])
  elif MODEL_TYPE == "pointnet": 
    training_data = DATALOADER_TYPES[MODEL_TYPE](trainingfiles, target_np=training_settings["pointnet_points"])
    test_data = DATALOADER_TYPES[MODEL_TYPE](testfiles, target_np=training_settings["pointnet_points"])
  else: 
    training_data = DATALOADER_TYPES[MODEL_TYPE](trainingfiles)
    test_data = DATALOADER_TYPES[MODEL_TYPE](testfiles)
  print(f"Training data size: {len(training_data)}; Test data size: {len(test_data)}; Batch size: {BATCH_SIZE}; Worker number: {WORKER_NR}")

  classifier = get_model(MODEL_TYPE, training_settings["class_nr"])
  print(f"Classifier: {classifier}")

  # if (args.verbose > 0) or (not args.production): 
  #   if MODEL_TYPE in ["pointnet", "pointnet2", "dgcnn", "paconv"]:
  #     tensorboard_writer.add_graph(classifier, torch.randn(1, 3, INPUT_POINTS))
  #   elif MODEL_TYPE in ["voxnet", "deeprank", "gnina"]:
  #     tensorboard_writer.add_graph(classifier, torch.randn(1, 1, 32, 32, 32))
  #   elif MODEL_TYPE in ["resnet", "convnext", "convnext_iso", "swintrans", "ViT"]:
  #     tensorboard_writer.add_graph(classifier, torch.randn(1, 1, 128, 128))

  # Use KaiMing He's initialization
  c = 0
  for m in classifier.modules():
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
      print(f"Init Conv Layer {c}")
      nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
      if m.bias is not None:
        nn.init.constant_(m.bias, 0)
      c += 1
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
      print(f"Init BatchNorm Layer {c}")
      nn.init.constant_(m.weight, 1)
      nn.init.constant_(m.bias, 0)
      c += 1
    elif isinstance(m, nn.Linear):
      print(f"Init Linear Layer {c}")
      nn.init.normal_(m.weight, 0, 0.1)
      if m.bias is not None:
        nn.init.constant_(m.bias, 0)
      c += 1

  if training_settings["pretrained"] and len(training_settings["pretrained"]) > 0:
    classifier.load_state_dict(torch.load(training_settings["pretrained"]))
  if USECUDA:
    classifier.cuda()

  if training_settings["optimizer"] not in OPTIMIZERS:
    print(f"Unexpected optimizer {training_settings['optimizer']}; Using Adam by default. ")
  optimizer = OPTIMIZERS.get(training_settings["optimizer"], optim.Adam)(classifier.parameters(), lr=training_settings["lr_init"], betas=(0.9, 0.999))
  # Other choices (SGD not performing well)
  # optimizer = optim.SGD(classifier.parameters(), lr=training_settings["learning_rate"], momentum=0.5)

  # The loss function in the original training
  criterion = LOSS_FUNCTIONS.get(training_settings["loss_function"], nn.CrossEntropyLoss)()
  print(f"Optimization function: {optimizer}")
  print(f"Loss function: {criterion}")

  # Learning rate scheduler
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=training_settings["lr_decay_steps"], gamma=training_settings["lr_decay_rate"])
  
  print(f"Number of parameters: {sum([p.numel() for p in classifier.parameters()])}")
  for epoch in range(0, EPOCH_NR): 
    st = time.perf_counter()
    st_training = time.perf_counter()
    if (epoch < START_EPOCH): 
      print(f"Skip the epoch {epoch}/{START_EPOCH} ...")
      scheduler.step()
      print(f"Epoch {epoch} took {time.perf_counter() - st_training:6.2f} seconds to train. Current learning rate: {get_lr(optimizer):.6f}. ")
      continue
    message = f" Running the epoch {epoch:>4d}/{EPOCH_NR:<4d} at {time.ctime()} "
    print(f"{message:#^80}")
    batch_nr = (len(training_data) + BATCH_SIZE - 1) // BATCH_SIZE
    for batch_idx, batch in enumerate(training_data.mini_batches(batch_size=BATCH_SIZE, process_nr=WORKER_NR)):
      retrieval_time = time.perf_counter() - st
      train_data, train_label = batch
      if len(train_label) != BATCH_SIZE:
        print(f"Skip the batch {batch_idx} due to the small batch size {len(train_label)}")
        continue
      # print(train_data.shape, train_label.shape)
      if isinstance(training_data, dataloader.CoordDataset) or isinstance(training_data, dataloader.SurfDataset):
        train_data = train_data.transpose(2, 1) 
      
      # print(train_data.shape)  # For test purpose
      st_tr = time.perf_counter()
      if USECUDA:
        train_data, train_label = train_data.cuda(), train_label.cuda()
      optimizer.zero_grad()
      classifier = classifier.train()
      pred = classifier(train_data)
      if isinstance(classifier, PointNetCls) or isinstance(pred, tuple):
        pred = pred[0]

      # Get the logit if the huggingface models is used 
      if isinstance(pred, transformers.file_utils.ModelOutput): 
        pred = pred.logits
      # print(pred.shape, train_label.shape)  # For test purpose

      loss = criterion(pred, train_label)
      loss.backward()
      optimizer.step()
      train_time = time.perf_counter() - st_tr
      
      if batch_idx % (batch_nr // 8) == 0: 
        loss_on_train, accuracy_on_train = test_model(classifier, training_data, criterion, 1024, BATCH_SIZE, USECUDA, WORKER_NR)
        loss_on_test, accuracy_on_test = test_model(classifier, test_data, criterion, 1024, BATCH_SIZE, USECUDA, WORKER_NR)
        jobmsg = f"Processing the block {batch_idx:>5d}/{batch_nr:<5d}; Loss: {loss_on_test:>6.4f}/{loss_on_train:<6.4f}; Accuracy: {accuracy_on_test:>6.4f}/{accuracy_on_train:<6.4f}; Time: {retrieval_time:>6.4f}/{train_time:<6.4f}; "
        if batch_idx > 0: 
          time_left = (time.perf_counter() - st_training) / (batch_idx + 1) * (batch_nr - batch_idx)
          jobmsg += f"Time-left: {time_left:5.0f}s; "
        print(jobmsg)

        if (args.verbose > 0) or (not args.production): 
          tensorboard_writer.add_scalar(f"Accuracy/tr_onfly", accuracy_on_train, epoch * batch_nr + batch_idx)
          tensorboard_writer.add_scalar(f"Accuracy/te_onfly", accuracy_on_test, epoch * batch_nr + batch_idx)
          parmset = list(classifier.named_parameters())
          for idx, p in enumerate(parmset):
            pname = p[0]
            p = p[1]
            if p.grad is not None:
              grad_norm = p.grad.norm()
              tensorboard_writer.add_scalar(f"global/grad{idx}_{pname}", grad_norm, epoch * batch_nr + batch_idx)
              tensorboard_writer.add_histogram(f"global/gradhist{idx}_{pname}", p.grad.cpu().detach().numpy(), epoch * batch_nr + batch_idx)
          image = match_data(pred, train_label)
          tensorboard_writer.add_image(f"match/match", image, epoch * batch_nr + batch_idx)
          tensorboard_writer.add_histogram("dist/pred", torch.argmax(pred, dim=1), epoch * batch_nr + batch_idx, bins=np.arange(-0.5, training_settings["class_nr"] + 0.5, 1))
          tensorboard_writer.add_histogram("dist/label", train_label, epoch * batch_nr + batch_idx, bins=np.arange(-0.5, training_settings["class_nr"] + 0.5, 1))

      st = time.perf_counter()

    # Test the model on both the training set and the test set
    scheduler.step()
    print(f"Epoch {epoch} took {time.perf_counter() - st_training:6.2f} seconds to train. Current learning rate: {get_lr(optimizer):.6f}. ")
    loss_on_train, accuracy_on_train = test_model(classifier, training_data, criterion, training_settings["test_number"], BATCH_SIZE, USECUDA, WORKER_NR)
    loss_on_test, accuracy_on_test = test_model(classifier, test_data, criterion, training_settings["test_number"], BATCH_SIZE, USECUDA, WORKER_NR)
    print(f"Checking the Performance on Loss: {loss_on_test}/{loss_on_train}; Accuracy: {accuracy_on_test}/{accuracy_on_train} at {time.ctime()}")
    current_lr = get_lr(optimizer)
    
    # Save the model  
    modelfile_output = os.path.join(os.path.abspath(training_settings["output_folder"]), f"{MODEL_TYPE}_{epoch}.pth")
    print(f"Saving the model to {modelfile_output} ...")
    torch.save(classifier.state_dict(), modelfile_output)
    
    if (args.verbose > 0) or (not args.production):
      tensorboard_writer.add_scalar("Loss/Train", loss_on_train, epoch)
      tensorboard_writer.add_scalar("Accuracy/Train", accuracy_on_train, epoch)
      tensorboard_writer.add_scalar("Loss/Test", loss_on_test, epoch)
      tensorboard_writer.add_scalar("Accuracy/Test", accuracy_on_test, epoch)
      tensorboard_writer.add_scalar("LearningRate", current_lr, epoch)

    # Save the performance to a HDF5 file
    with feater.io.hdffile(os.path.join(training_settings["output_folder"], "performance.h5") , "a") as hdffile:
      utils.update_hdf_by_slice(hdffile, "loss_train", np.array([loss_on_train], dtype=np.float64), np.s_[epoch:epoch+1], dtype=np.float64, maxshape=(None, )) 
      utils.update_hdf_by_slice(hdffile, "loss_test", np.array([loss_on_test], dtype=np.float64), np.s_[epoch:epoch+1], dtype=np.float64, maxshape=(None, ))
      utils.update_hdf_by_slice(hdffile, "accuracy_train", np.array([accuracy_on_train], dtype=np.float64), np.s_[epoch:epoch+1], dtype=np.float64, maxshape=(None, ))
      utils.update_hdf_by_slice(hdffile, "accuracy_test", np.array([accuracy_on_test], dtype=np.float64), np.s_[epoch:epoch+1], dtype=np.float64, maxshape=(None, ))

    # Early stopping on low accuracy (model did not learn anything at all)
    # Potential incorrect initialization or hyperparameters
    if (epoch > 0.25 * EPOCH_NR) and (accuracy_on_train < 0.1): 
      print(f"Early stopping at epoch {epoch} due to the extreme low accuracy on the training set (accuracy < 0.1)")
      print(f"Please check if hyperparameters are set correctly or the initialization of the model is correct. ")
      break
    # Early stopping on high accuracy to avoid overfitting
    if accuracy_on_test > 0.995: 
      print(f"Early stopping at epoch {epoch} due to the high accuracy on the test set (accuracy > 0.995)")
      break

def console_interface():
  global tensorboard_writer
  global args
  args = parse_args()
  SETTINGS = vars(args)
  _SETTINGS = json.dumps(SETTINGS, indent=2)
  print("Arguments of this training:")
  print(_SETTINGS)
  with open(os.path.join(SETTINGS["output_folder"], "settings.json"), "w") as f:
    f.write(_SETTINGS)

  if (args.verbose > 0) or (not args.production): 
    if not os.path.exists(os.path.join(SETTINGS["output_folder"], "tensorboard")): 
      os.makedirs(os.path.join(SETTINGS["output_folder"], "tensorboard")) 
    tensorboard_writer = SummaryWriter(os.path.join(SETTINGS["output_folder"], "tensorboard"))

  perform_training(SETTINGS)

if __name__ == "__main__":
  """
  Train with 
  
  Train with pointnet-coord: 
  python train_voxnet.py -m pointnet -train /diskssd/yzhang/FEater_Minisets/miniset_200/te_coord.txt -test /Weiss/FEater_Dual_PDBHDF/te.txt -o /tmp/testpointnet -w 24 -e 120 -b 128 -lr 0.001 --lr-decay-steps 30 --lr-decay-rate 0.5 --data-type dual --pointnet_points 48

  Train with pointnet-surf: 
  python train_voxnet.py -m pointnet -train /diskssd/yzhang/FEater_Minisets/miniset_200/te_surf.txt -test /Weiss/FEater_Dual_SURF/te.txt -o /tmp/testpointnet -w 24 -e 120 -b 128 -lr 0.001 --lr-decay-steps 30 --lr-decay-rate 0.5 --data-type dual --pointnet_points 2000 --dataloader-type surface

  Training with VoxNet
  python train_voxnet.py -m voxnet -train /Weiss/FEater_Data/FEater_Minisets/tr_single_vox.txt -test /Weiss/FEater_Single_VOX/te.txt -o /tmp/testvoxnet -w 24 -e 120 -b 128 -lr 0.001 --lr-decay-steps 30 --lr-decay-rate 0.5 --data-type single

  Training with ResNet
  python train_voxnet.py -m resnet -train /Weiss/FEater_Data/FEater_Minisets/tr_dual_hilb.txt -test /Weiss/FEater_Dual_HILB/te.txt -o /tmp/testresnet -w 24 -e 120 -b 128 -lr 0.005 --lr-decay-steps 30 --lr-decay-rate 0.5 --data-type dual
  """

  args = parse_args()
  SETTINGS = vars(args)
  _SETTINGS = json.dumps(SETTINGS, indent=2)
  print("Settings of this training:")
  print(_SETTINGS)

  with open(os.path.join(SETTINGS["output_folder"], "settings.json"), "w") as f:
    f.write(_SETTINGS)
  
  if (args.verbose > 0) or (not args.production): 
    if not os.path.exists(os.path.join(SETTINGS["output_folder"], "tensorboard")): 
      os.makedirs(os.path.join(SETTINGS["output_folder"], "tensorboard")) 
    tensorboard_writer = SummaryWriter(os.path.join(SETTINGS["output_folder"], "tensorboard"))

  perform_training(SETTINGS)

