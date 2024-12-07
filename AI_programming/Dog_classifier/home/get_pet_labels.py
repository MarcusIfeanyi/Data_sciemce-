#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/get_pet_labels.py
#                                                                             
# PROGRAMMER: Agbaji Marcus Ifeanyi
# DATE CREATED:  12-01-2023                                
# REVISED DATE:  14/02/2023
# PURPOSE:To Create the function get_pet_labels that creates the pet labels from 
#          the image's filename. This function inputs: 
#           - The Image Folder as image_dir within get_pet_labels function and 
#             as in_arg.dir for the function call within the main function. 
#          This function creates and returns the results dictionary as results_dic
#          within get_pet_labels function and as results within main. 
#          The results_dic dictionary has a 'key' that's the image filename and
#          a 'value' that's a list. This list will contain the following item
#          at index 0 : pet image label (string).
#
##
# Imports python modules
from os import listdir
import string 


def get_pet_labels(image_dir):
    """
    Creates a dictionary of pet labels (results_dic) based upon the filenames 
    of the image files. These pet image labels are used to check the accuracy 
    of the labels that are returned by the classifier function, since the 
    filenames of the images contain the true identity of the pet in the image.
    Be sure to format the pet labels so that they are in all lower case letters
    and with leading and trailing whitespace characters stripped from them.
    (ex. filename = 'Boston_terrier_02259.jpg' Pet label = 'boston terrier')
    Parameters:
     image_dir - The (full) path to the folder of images that are to be
                 classified by the classifier function (string)
    Returns:
      results_dic - Dictionary with 'key' as image filename and 'value' as a 
      List. The list contains for following item:
         index 0 = pet image label (string)
    """

    results_dic = {}
    list_pet = listdir(image_dir)
    lower_list_pets= []
 
    pet_labels =[]
    
    for i in list_pet:
        
        lower_list_pets.append(i)
    for jpg in lower_list_pets:
        if jpg[0] != '.': 
            labels = jpg.split('_')
            pet_label = ' '.join([label.lower().strip() for label in labels if label.isalpha()])
#             pet_label = " "
#             for label in labels:
#                 if label.isalpha():
#                     pet_label += (label + " ").lower()
#             pet_label.strip()
            pet_labels.append(pet_label)
            

    for i, img in enumerate(lower_list_pets):
        if img not in results_dic:
            results_dic[img] = [pet_labels[i].strip()]
        else:
            print("** Warning: Duplicate files exist in directory:",result_dic[img])
            
    return (results_dic)   

if __name__ == "__main__":
    results_dic = {}
    list_pet = listdir("pet_images/")
    lower_list_pets= []
 
    pet_labels =[]
    
    for i in list_pet:
        
        lower_list_pets.append(i)
    for jpg in lower_list_pets:
        if jpg[0] != '.': 
            labels = jpg.split('_')
            pet_label = ' '.join([label.lower().strip() for label in labels if label.isalpha()])
#             pet_label = " "
#             for label in labels:
#                 if label.isalpha():
#                     pet_label += (label + " ").lower()
#             pet_label.strip()
            pet_labels.append(pet_label)
            

    for i, img in enumerate(lower_list_pets):
        if img not in results_dic:
            results_dic[img] = [pet_labels[i].strip()]
        else:
            print("** Warning: Duplicate files exist in directory:",result_dic[img])
            
    print(results_dic)

# Creates empty dictionary named results_dic


# results_dic = dict()

# # Determines number of items in dictionary
# items_in_dic = len(results_dic)
# print("\nEmpty Dictionary results_dic - n items=", items_in_dic)

# # Adds new key-value pairs to dictionary ONLY when key doesn't already exist. This dictionary's value is
# # a List that contains only one item - the pet image label
# filenames = ["beagle_0239.jpg", "Boston_terrier_02259.jpg"]
# pet_labels = ["beagle", "boston terrier"]
# for idx in range(0, len(filenames), 1):
#     if filenames[idx] not in results_dic:
#          results_dic[filenames[idx]] = [pet_labels[idx]]
#     else:
#          print("** Warning: Key=", filenames[idx], 
#                "already exists in results_dic with value =", 
#                results_dic[filenames[idx]])

# #Iterating through a dictionary printing all keys & their associated values
# print("\nPrinting all key-value pairs in dictionary results_dic:")
# for key in results_dic:
#     print("Filename=", key, "   Pet Label=", results_dic[key][0])

# print(results_dic)
# # l = []
# for f in filenames:
#     l = f.split('_')
#     print(l)

#       TODO: THIS WAS CODDED BY AGBAJI MARCUS IFEANYI
# results_dic = {}
# list_pet = listdir('pet_images/')
# lower_list_pets= []
 
# pet_labels =[]

# for i in list_pet:
    
#     lower_list_pets.append(i)
# for jpg in lower_list_pets:
#     labels = jpg.split('_')
#     pet_label = " "
#     for label in labels:
#         if label.isalpha():
#             pet_label += (label + " ").lower()
#     pet_label.strip()
#     pet_labels.append(pet_label)

# for i, img in enumerate(lower_list_pets):
#     if img not in results_dic:
#         results_dic[img] = [pet_labels[i]]
        
# print(results_dic)

# count = 0
# for key in results_dic:
#     count += 1
# print(count)
    


