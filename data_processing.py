
import csv


NUM_LOGS = 10 #number of log files


#skill tag dictionary to map skill id
skills = dict()



def write_data(file_name, skill_tags, answers, num_actions):

    
    #print("num_actions : {}".format(num_actions))


#    print("skill_tags size: {}".format(len(skill_tags)))
    if(num_actions < 3):

        return
    print("skill_tags size: {}".format(len(skill_tags)))
  
    print("number of actions: {}".format(num_actions))


    with open(file_name, 'a') as output:
        
        output.write(str(num_actions) + '\n')

        for i in range(0, num_actions):

            if(i != num_actions-1):

                # print("i=" +str(i));
                
                output.write(str(skill_tags[i]) + ',')
            
            else:
                
                output.write(str(skill_tags[i]) + '\n')


        for i in range(0, num_actions):

            if(i != num_actions-1):
                
                output.write(str(answers[i]) + ',')
            
            else:
                
                output.write(str(answers[i]) + '\n\n\n')





if __name__ == "__main__":

    
    for i in range(0,NUM_LOGS):

        file_name = "student_log_{}.csv".format(i+1)

        with open(file_name, 'r') as log:
            
            if i == 0: 
                cur_id = -1 # current student id

                num_actions = 0 #number of actions for current student
                
                skill_tags = [] #the skill tags of each problems

                answers = []   #the answers of current students
            
            
            reader = csv.DictReader(log)
            
            
            for row in reader:
                
                if(int(row["ITEST_id"]) != cur_id): #new students
                
                    print("write student {}".format(cur_id))

                    write_data("data1.csv", skill_tags, answers, num_actions)
                    
                    cur_id = int(row["ITEST_id"])

                    num_actions = int(row["NumActions"])

                    skill_tags = []

                    answers = []


                #check whether the skills dict has the skill, if not, add the skill as new key
                if(not (str(row["skill"]) in skills )):

                        skills[str(row["skill"])] = len(skills) + 1


                skill_tags.append(skills[row["skill"]]);
            
                      
                
                #print(len(skill_tags))
                #print("cur_id: {}".format(cur_id))
                
                answers.append(int(row["correct"]))




















