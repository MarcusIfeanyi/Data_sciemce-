from get_pet_labels import get_pet_labels
dog_names_dic = {}
dog_names_list = []
is_dog_p = None
is_dog_c = None
results_dic = get_pet_labels("pet_images/")
print(results_dic)
with open('dognames.txt','r') as dog_file:
    for line in dog_file:

        if (line not in dog_names_list) or (line not in dog_names_dic):
            dog_names_list.append(line.lower().strip())
            dog_names_dic[line] = 1
        else:
            print('Warning, file already exist',dog_names_dic[line])
           
print("this is messed up}")
print(dog_names_list)
if 'walker hound\n' in dog_names_list:
    print( 'THIS IS ME')
else: print('this is not me')

for key in results_dic:
    if results_dic[key][0] in dog_names_list:
        is_dog_p = 1
    else:
        is_dog_p = 0
    if results_dic[key][1] in  dog_names_list:
        is_dog_c = 1
    else:
        is_dog_c = 0
                      
        results_dic[key].extend([is_dog_p, is_dog_c])
        
print(result_dic)
                      
      