""" Pytorch datasets. """
from dermosxai import data

class IAD():
  """ Interactive Atlas of Dermoscopy (IAD) dataset.
  
  Arguments:
    split (string): String with the split to use: "train", "val", "test".
    transform ()
    return_attributes (bool): Whether attributes should be returned along with images and labels
    one_per_patient
  
  Returns:
    image: An image fromt this patient
    label: A single digit, label of the image
    (optionally) attributes : encoded attributes for this example
  """
   def __init__(self, split='train' transform=None, return_attributes=False):
       # Load images
       images, labels = data.get_IAD_images()
	
	# Load attributes
	if return_attributes:
	  attributes, attribonfig --global user.email "you@example.com"
  git config --global user.name "Your Name"ute_labels = self.load_attributes
	  
       # Subsample to right split
       if split== 'train':
     	 images  = images[train_split]
     	 if return_attributes:
     	   
       else:   
       
       # Make sure there is only one image per patient
       if one_per_patient: # diff images for same patient usually look the same
       	images = images[:, -1] # just pick the last image for each subject
       else:
       	images = images.ravel(axis=1)
       	labels = # copy the labels accorsingly
       	if attributes:
       	  ssksksks
         
        
        
   def __getitem__(self, i):
     if self.return_attributes:
       example = (self.images[i], self.labels[i], self.attributes[i])
     else:
       example = (self.images[i], self.labels[i])
     return example
    
    def __len__(self):
      return len(self.images)
   
   
# class HAM1000()
#  def __init__():
#  	# Make sure I use the HAM1000 official valisatio nd test set.
 	
#  	# maybe sample the test set from the training set as we don't have access to the original test set.
      
