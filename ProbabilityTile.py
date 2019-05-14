# -*- coding: utf-8 -*-
import numpy as np

import one_hot_labels

class ProbabilityTile:
    
    """ProbabilityTile
    
    Class for aggregating multiple predicted patches of a tile into a single tile 
    prediction. sum_matrix is adequate for decoding into an RGB label image as it 
    uses the highest summed probability for each pixel. If an averaged probability 
    mask with values [0, 1] are needed, use get_prob_matrix().
    
    """
    
    NB_LABELS = one_hot_labels.NB_LABELS
    
    def __init__(self, height, length):
        self.height = height
        self.length = length
        # Matrix tracking the summed probabilites for each pixel
        self.sum_matrix = np.zeros((height, length, self.NB_LABELS))
        # Matrix tracking the number of patches merged for each pixel
        self.num_matrix = np.zeros((height, length), dtype = np.uint16) 
        
        self._prob_updated = False
        self._prob_matrix = np.zeros((height, length, self.NB_LABELS))
        
    def _calc_prob_matrix(self):
        """Calculate the internal average probability matrix by dividing sum."""
        with np.errstate(divide='ignore', invalid='ignore'): # Ignore divide by zero warnings
            self._prob_matrix = self.sum_matrix / self.num_matrix.reshape(
                    (self.height,self.length,1))
        self._prob_updated = True
        
    def merge_prob_patch(self, x, y, patch):
        """Merge Probability Patch
        Add a probability patch to this probability tile. Does not calculate 
        average probabilities.
        
        # Arguments
            x: Integer. X-coordinate of the input patch.
            y: Integer. Y-coordinate of the input patch.
            patch: 3D NumPy array. The probability patch to merge. The last
                channel must be a vector of NB_LABELS size containing probabilities
                summing to 1.
        """
        self._prob_updated = False
        self.sum_matrix[y:y+patch.shape[0], x:x+patch.shape[1], :] += patch
        self.num_matrix[y:y+patch.shape[0], x:x+patch.shape[1]] += 1
    
    def get_prob_matrix(self):
        """Get a probability matrix where the sum of probabilities in each pixel is 1."""
        if self._prob_updated:
            return self._prob_matrix
        self._calc_prob_matrix()
        return np.nan_to_num(self._prob_matrix.copy())
    
    def get_tile_label(self):
        """Get Tile Label
        Get an RGB label image of this ProbabilityTile where each pixel is
        classified as the class with highest probability.
        
        # Returns
            3D NumPy array in RGB.
        """
        return one_hot_labels.probability_decode(self.sum_matrix)
        