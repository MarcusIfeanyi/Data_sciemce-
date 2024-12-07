#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/adjust_results4_isadog.py
#                                                                             
# PROGRAMMER: AGBAJI MARCUS IFEANYI
# DATE CREATED: 16TH jenuary 2023             
# REVISED DATE: 14/02/2023 
# PURPOSE: To Create a function adjust_results4_isadog that adjusts the results 
#          dictionary to indicate whether or not the pet image label is of-a-dog, 
#          and to indicate whether or not the classifier image label is of-a-dog.
#          All dog labels from both the pet images and the classifier function
#          will be found in the dognames.txt file. We recommend reading all the
#          dog names in dognames.txt into a dictionary where the 'key' is the 
#          dog name (from dognames.txt) and the 'value' is one. If a label is 
#          found to exist within this dictionary of dog names then the label 
#          is of-a-dog, otherwise the label isn't of a dog. Alternatively one 
#          could also read all the dog names into a list and then if the label
#          is found to exist within this list - the label is of-a-dog, otherwise
#          the label isn't of a dog. 
#         This function inputs:
#            -The results dictionary as results_dic within adjust_results4_isadog 
#             function and results for the function call within main.
#            -The text file with dog names as dogfile within adjust_results4_isadog
#             function and in_arg.dogfile for the function call within main. 
#           This function uses the extend function to add items to the list 
#           that's the 'value' of the results dictionary. You will be adding the
#           whether or not the pet image label is of-a-dog as the item at index
#           3 of the list and whether or not the classifier label is of-a-dog as
#           the item at index 4 of the list. Note we recommend setting the values
#           at indices 3 & 4 to 1 when the label is of-a-dog and to 0 when the 
#           label isn't a dog.
#
#

def adjust_results4_isadog(results_dic, dogfile):
    """
    Adjusts the results dictionary to determine if classifier correctly 
    classified images 'as a dog' or 'not a dog' especially when not a match. 
    Demonstrates if model architecture correctly classifies dog images even if
    it gets dog breed wrong (not a match).
    Parameters:
      results_dic - Dictionary with 'key' as image filename and 'value' as a 
                    List. Where the list will contain the following items: 
                  index 0 = pet image label (string)
                  index 1 = classifier label (string)
                  index 2 = 1/0 (int)  where 1 = match between pet image
                    and classifer labels and 0 = no match between labels
                ------ where index 3 & index 4 are added by this function -----
                 NEW - index 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and 
                            0 = pet Image 'is-NOT-a' dog. 
                 NEW - index 4 = 1/0 (int)  where 1 = Classifier classifies image 
                            'as-a' dog and 0 = Classifier classifies image  
                            'as-NOT-a' dog.
     dogfile - A text file that contains names of all dogs from the classifier
               function and dog names from the pet image files. This file has 
               one dog name per line dog names are all in lowercase with 
               spaces separating the distinct words of the dog name. Dog names
               from the classifier function can be a string of dog names separated
               by commas when a particular breed of dog has multiple dog names 
               associated with that breed (ex. maltese dog, maltese terrier, 
               maltese) (string - indicates text file's filename)
    Returns:
           None - results_dic is mutable data type so no return needed.
    """  
    dog_names_dic = {}
    dog_names_list = []
#     is_dog_p = None
#     is_dog_c = None    
    with open(dogfile,'r') as dog_file:
        for line in dog_file:
       
            if (line not in dog_names_list): #or (line not in dog_names_dic):
                dog_names_list.append(line.lower().strip())
                dog_names_dic[line.lower().strip()] = 1
            else:
                print('Warning, file already exist',dog_names_dic[line])
            
            
    for key, value in results_dic.items():
                      
        if value[0] in dog_names_dic:
            is_dog_p = 1
        else:
            is_dog_p = 0
        if value[1] in  dog_names_dic:
            is_dog_c = 1
        else:
             is_dog_c = 0
                      
        results_dic[key].extend([is_dog_p, is_dog_c])
        
#     return (result_dic)

if __name__ == "__main__":
    
    from calculates_results_stats import calculates_results_stats
    from classify_images import classify_images
    from get_pet_labels import get_pet_labels
    
    
#     results_dic = {}
#     list_pet = listdir("pet_images/")
#     lower_list_pets= []
 
#     pet_labels =[]

#     for i in list_pet:
        
#         lower_list_pets.append(i)
#     for jpg in lower_list_pets:
#         labels = jpg.split('_')
#         pet_label = " "
#         for label in labels:
#             if label.isalpha():
#                 pet_label += (label + " ").lower()
#         pet_label.strip()
#         pet_labels.append(pet_label)

#     for i, img in enumerate(lower_list_pets):
#         if img not in results_dic:
#             results_dic[img] = [pet_labels[i]]
#         else:
#             print("** Warning: Duplicate files exist in directory:",result_dic[img])
    r_dict = get_pet_labels("pet_images/")
    classify_images("pet_images", r_dict, "resnet")
    adjust_results4_isadog(r_dict, "dognames.txt" )
    result = calculates_results_stats(r_dict)
    for key in result:
        if "pct" in key:
            print(key, result[key])
    
    
                      
                       
        
            
                
            
        
    
