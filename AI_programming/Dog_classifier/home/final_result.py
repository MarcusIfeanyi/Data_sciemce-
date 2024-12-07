print("\n{:20}: {:3d}".format('# TOTAL IMAGES', 4))    
print("{:20}: {:3d}".format('# DOG IMAGES', 2))
print("{:20}: {:3d}\n".format('# NOT-A-DOG-IMAGES', 2))
    
    
                   
print("  {:20}: {:5} {:3} {:3} {:3}".format("CNN Model Architecture", "% Not-a-Dog Correct", "% Dogs Correct", "% Breeds Correct", "% Match Labels"))
print("  {:22}: {:>15} {:>15} {:>15} {:>15}".format("ResNet","100.0%","100.0%",'100.0%', '100.0%'))
print("  {:22}: {:>15} {:>15} {:>15} {:>15}".format("AlexNet","100.0%","100.0%",'100.0%', '75.0%'))
print("  {:22}: {:>15} {:>15} {:>15} {:>15}\n".format("ResNet","100.0%","100.0%",'100.0%', '75.0%'))
print(" *** SUMMARY OF THE EXECISE***\n Form the execise the best model is the ResNet model. It classified accurately all the dog, breeds and not dogs and matches" )
print('''It is observed that the VGG and the AlexNet classified all accurately except  "label match" which it incorrectly classified''')




          