#
#  Takes in a txt file from the Tecan output and calculates doubling time and lag time for each well, and plots the curves
#
###  this uses two averaging smoothes and a backwards smooth to make curves from the spectramax look ok.  plotting is possible at the bottom

#data_file_name = "/Users/jroop/Downloads/no_shake_test.txt"



import numpy as np
#from pylab import *

import matplotlib.pyplot as plt
import matplotlib.colors as colors

#colors.Colormap('bone')


from scipy import stats
import sys


#in_wells = {'Sbay CBS7001_gal' : ['A01', 'A02'], 'BY (YHL068)' : ['B01', 'B02'], 'YPS128' : ['C01', 'C02'], 'SK1' : ['D01', 'D02'], 'Y9' : ['E01', 'E02'], 'D1853' : ['F01', 'F02'], 'D1788' : ['G01', 'G02'], 'D1373' : ['H01', 'H02']}

#in_wells = {'Sbay CBS7001_gal' : ['A08', 'A09', 'A10'], 'BY (YHL068)' : ['B08', 'B09', 'B10'], 'YPS128' : ['C08', 'C09', 'C10'], 'SK1' : ['D08', 'D09', 'D10'], 'Y9' : ['E08', 'E09', 'E10'], 'D1853' : ['F08', 'F09', 'F10'], 'D1788' : ['G08', 'G09', 'G10'], 'D1373' : ['H08', 'H09', 'H10']}

#in_wells = {'scer' : ['A10', 'A11', 'A12'], 'scer ura': ['B10', 'B11', 'B12'], 'spar' : ['C10', 'C11', 'C12'], 'smik' : ['D10', 'D11', 'D12'], 'sarb' : ['E10', 'E11', 'E12'], 'sbay' : ['F10', 'F11', 'F12'], 'ncas' : ['H10', 'H11', 'H12']}
#in_wells = {'Sbay': ['F04','F05', 'F06'],'Scer' : ['A04', 'A05', 'A06'], 'Spar': ['C04', 'C05', 'C06'], 'Smik': ['D04', 'D05', 'D06'], 'Ncas' : ['H04', 'H05', 'H06']}
#in_wells = {'Sbay': ['F01','F02', 'F03'],'Scer' : ['A01', 'A02', 'A03'], 'Spar': ['C01', 'C02', 'C03'], 'Smik': ['D01', 'D02', 'D03'], 'Ncas' : ['H01', 'H02', 'H03']}

#in_wells = {'Sc-gal': ['A03','A04', 'G03', 'G04'],'sc-glu' : ['A01', 'A02', 'G01', 'G02'], 'mig1_glu' : ['C01', 'C02', 'D01', 'D02'],'mig1 gal' : ['C03', 'C04', 'D03', 'D04'],}#, 'Sb-gal' : ['B03','B04', 'H03', 'H04'], 'Sb-glu' : ['B01','B02', 'H01', 'H02']}

in_wells = {'sb-mix' : ['B10', 'B11', 'B12', 'H10', 'H11', 'H12'], 'sc_mig1 mix' : ['C10', 'C11', 'C12', 'D10', 'D11', 'D12'], 'sc mix' : ['A10','A11', 'A12', 'G10', 'G11', 'G12']}#

#, 'Sb-glu' : ['B01','B02', 'H01', 'H02']}



#in_wells = {'Sc-gal': ['A03','A04', 'G03', 'G04'],'sc-glu' : ['A01', 'A02', 'G01', 'G02']}

#in_wells= {'sc-mix' : ['A10', 'A11', 'A12', 'G10', 'G11', 'G12'],'mig1 mix' : ['C10', 'C11', 'C12', 'D10', 'D11', 'D12']}



#in_wells = {'sbay' : ['F01', 'F02', 'F03'], 'sbay-mixed' : ['F10', 'F11', 'F12']}

#in_wells = {'sbay glu' : ['A07', 'A08', 'A09'], 'sbay gal': ['H07', 'H08', 'H09'], 'sbay .5+1.5' : ['D07', 'D08', 'D09'], 'sbay  .25+1.75' : ['E07', 'E08', 'E09'], 'sbay .125+1.875' : ['F07', 'F08', 'F09']}

#in_wells = {'scer glu' : ['A04', 'A05', 'A06'], 'scer gal': ['H04', 'H05'], 'scer .5+1.5' : ['D04', 'D05', 'D06'], 'scer  .25+1.75' : ['E04', 'E05', 'E06'], 'scer 1+1' : ['C04', 'C05', 'C06']}

#in_wells = {'sbay glu' : ['A01', 'A02', 'A03'], 'sbay gal': ['H01', 'H02', 'H03'], 'sbay .5+1.5' : ['D01', 'D02', 'D03'], 'sbay  .25+1.75' : ['E01', 'E02', 'E03'], 'sbay 1+1' : ['C01', 'C02', 'C03']}

#in_wells = {'H04' : ['H04'], 'H05' : ['H05'], 'H06' : ['H06']}

#in_wells = {'GAL1_RH_Sbay only' : ['F08', 'F09'], 'GAL1_RH_Scer only': ['G09', 'G08']}


legend = 'y' ### if you want a legend or not

#   , 'Scer + .25%' : ['B06', 'B07', 'B08', 'B09'], 'Sbay + .25%' : ['A06', 'A07', 'A08', 'A09']}

if len(sys.argv) == 1:
   
   print "\n", 'USAGE:', 'python', 'this_program', '-machine(s or t)', '-run_time(hours)', 'data_file', "\n"

   sys.exit()


machine = str(sys.argv[1]).upper().strip()[1:]



print machine
run_time = float((sys.argv[2])[1:])
data_file_name = sys.argv[3]



data_file = open(data_file_name, "r")

def Get_AOC(smooth_well_dict):
   

    corrected_well_dict = {}
    AOC_dict = {}

    ## transforms the smoothened curve by 1) subtracting the initial value which corresponds to media. 2) correcting for growth at high density then takes the middle rienman sum to get the area under the curve.  

    for well in smooth_well_dict:
        corrected_well_dict[well] = []

        for timepoint in range(len(smooth_well_dict[well])):

           


            corrected_well_dict[well].append(smooth_well_dict[well][timepoint]+((smooth_well_dict[well][timepoint]**3)*.8324))  # the warringer correction                   


    min_dict = {}


    for well in corrected_well_dict:
       mins_list = []
       mins_list.append(corrected_well_dict[well][0])
       mins_list.append(corrected_well_dict[well][1])
       mins_list.append(corrected_well_dict[well][2])

       min_avg = np.array(mins_list).mean()

       min_dict[well] = min_avg


    for well in corrected_well_dict:
       aoc = 0


       for timepoint in range(len(corrected_well_dict[well]))[:-1]:

          

          val = ((((corrected_well_dict[well][timepoint] + corrected_well_dict[well][timepoint+1])*.5)-min_dict[well])*30)  # 30 for tecan timepoints  10 for spectramax
          
          if val < 0:
             val = 0

          aoc+=val



       AOC_dict[well] = aoc


    return AOC_dict



def Smooth_data(well_dict):
    ## takes in a dictionary where keys are wells and all timepoint values are in a list and as the value and averages every point between the two points on either side
    ## also, if a point is less than the point before it, (t+1 > t) it will set it equal to the point before, t+1 = t

   smooth_dict = {}
   w = 'B04'

#   print well_dict[w]


   #plot(well_dict[w], color='blue')

   for well in well_dict:  ## first smooth
        
        
        for timepoint in range(len(well_dict[well][:-1])):  ## does not aveage at the first or last as this is impossible
            
            if timepoint == 0:
                smooth_values = [well_dict[well][timepoint]]
            else:
            
                smooth_values.append(np.array([well_dict[well][(timepoint-1)], well_dict[well][(timepoint)], well_dict[well][(timepoint+1)]]).mean())
            
        smooth_dict[well] = smooth_values

  # plot(smooth_dict[w])

 
   for well in smooth_dict:

      first = 'y'
      for timepoint in range(len(smooth_dict[well])):
         if first == 'y':
            first = 'n'
            continue


         if smooth_dict[well][timepoint] < smooth_dict[well][timepoint-1]:
#            print smooth_dict[well][timepoint], timepoint,
#            print smooth_dict[well][timepoint-1], timepoint-1


            smooth_dict[well][timepoint] = smooth_dict[well][timepoint-1]

       
 #  plot(smooth_dict[w])

   smoother_dict = {}

   for well in smooth_dict:  ## second smooth    

        for timepoint in range(len(smooth_dict[well][:-1])):  ## does not aveage at the first or last as this is impossible                                                      
#            print timepoint
            if timepoint == 0:
                smooth_values = [smooth_dict[well][timepoint]]
            else:

                smooth_values.append(np.array([smooth_dict[well][(timepoint-1)], smooth_dict[well][(timepoint)], smooth_dict[well][(timepoint+1)]]).mean())

        smoother_dict[well] = smooth_values


 #  print smoother_dict['A01']

#   plot(smoother_dict[w], color='yellow')

#   show()



   return smoother_dict


def Get_efficiency(smooth_well_dict):
    
    high_std_dev = []
    std_cutoff = .01
    efficiency_dict = {}

    all_init = []

    for well in smooth_well_dict:
       all_init.append(smooth_well_dict[well][2])
       
    mean_init = sum(all_init) / (len(smooth_well_dict))   ## this gets the mean of the initial value across all wells


    print mean_init

    for well in smooth_well_dict:

        first_two_timepoints = np.array(smooth_well_dict[well][1:3])

        efficiency_dict[well] = np.array(smooth_well_dict[well][-2:]).mean() - mean_init

        if efficiency_dict[well] < 0:
            efficiency_dict[well] = 0

        if first_two_timepoints.std() > std_cutoff:

            high_std_dev.append([well, first_two_timepoints.std()])

#    if len(high_std_dev) > 0:
#        print "CAUTION these well have a standard deviation of initial 2 values > ", str(std_cutoff)
#        for item in high_std_dev:
#            print item

    return efficiency_dict



def Correct_curves(smooth_well_dict):

    corrected_well_dict = {}
    efficiency_dict = {}

    ## transforms the smoothened curve by 1) subtracting the initial value which corresponds to media. 2) log2 transforming and 3)  correcting for growth at high density 

    for well in smooth_well_dict:
        corrected_well_dict[well] = []

        for timepoint in range(len(smooth_well_dict[well])):
            
            corrected_well_dict[well].append(np.log(smooth_well_dict[well][timepoint]+((smooth_well_dict[well][timepoint]**3)*.8324)))  # the warringer correction

    return corrected_well_dict

def Get_double_time_and_lag(corrected_well_dict, machine, run_time):
    
    all_init = []

    for well in corrected_well_dict:
       all_init.append(corrected_well_dict[well][2])

    mean_init = sum(all_init) / (len(corrected_well_dict))   ## this gets the mean of the initial value across all wells from the corrected dict     



#    timepoint_interval = 10  ## the number of minutes between OD readings
    slope_and_lag_dict = {}
    
            
    for well in corrected_well_dict:

  #      w = 'A01'
 #       plot(corrected_well_dict[w])
#        show()


        all_slopes = []
        if machine == 'S':  ### for spectramax  ## 40 MINUTE INTEVALS!!
         timepoint_interval = 10  
         for timepoint in range(len(corrected_well_dict[well])-3):  ## or 12?


#            regress_data = stats.linregress([(timepoint)*timepoint_interval, (timepoint+2)*timepoint_interval, (timepoint+4)*timepoint_interval, (timepoint+6)*timepoint_interval, (timepoint+8)*timepoint_interval, (timepoint+10)*timepoint_interval, (timepoint+12)*timepoint_interval], [corrected_well_dict[well][timepoint], corrected_well_dict[well][timepoint+2], corrected_well_dict[well][timepoint+4], corrected_well_dict[well][timepoint+6], corrected_well_dict[well][timepoint+8], corrected_well_dict[well][timepoint+10], corrected_well_dict[well][timepoint+12]])
            regress_data = stats.linregress([(timepoint)*timepoint_interval, (timepoint+1)*timepoint_interval, (timepoint+2)*timepoint_interval, (timepoint+3)*timepoint_interval], [corrected_well_dict[well][timepoint], corrected_well_dict[well][timepoint+1], corrected_well_dict[well][timepoint+2], corrected_well_dict[well][timepoint+3]])


            all_slopes.append(regress_data[:2])  ## appends the slope of each linregress


## for TECAN

        if machine == 'T' or machine == 'TB':

         timepoint_interval = 30
         for timepoint in range(len(corrected_well_dict[well])-2):



            regress_data = stats.linregress([(timepoint)*timepoint_interval, (timepoint+1)*timepoint_interval, (timepoint+2)*timepoint_interval], [corrected_well_dict[well][timepoint], corrected_well_dict[well][timepoint+1], corrected_well_dict[well][timepoint+2]])




            all_slopes.append(regress_data[:2])  ## appends the slope of each linregress                           

            ## sort and take the biggest slope with the intercept ## 


        all_slopes.sort()
        all_slopes.reverse()


        highest_slope_mean = [((all_slopes[2][0] + all_slopes[3][0] + all_slopes[4][0] + all_slopes[5][0] + all_slopes[6][0]) / 5.0), ((all_slopes[2][1] + all_slopes[3][1] + all_slopes[4][1] + all_slopes[5][1] + all_slopes[6][1]) / 5.0)]  ## ignored fist 2, averges next 5

        doubling_time = (np.log(2) / highest_slope_mean[0]) / 60

        if doubling_time > run_time:
           doubling_time = run_time


 #       print highest_slope_mean[1]

#        lag_time = -((highest_slope_mean[1] - mean(corrected_well_dict[well][:2])) / highest_slope_mean[0]) / 60
        lag_time = (((corrected_well_dict[well][1] + corrected_well_dict[well][2])/2 - highest_slope_mean[1]) / highest_slope_mean[0]) / 60  ## for using the first value of every curve
 #       lag_time = ((mean_init - highest_slope_mean[1]) / highest_slope_mean[0]) / 60   ## for using the mean line as the initial

        if lag_time > run_time or doubling_time == run_time:
            lag_time = run_time

        slope_and_lag_dict[well] = [doubling_time, lag_time]

    return slope_and_lag_dict

def Print_output(efficiency_dict, slope_and_lag_dict, aoc_dict):

    all_wells = slope_and_lag_dict.keys()
    all_wells.sort()

    print "WELL", "\t", "DOUBLING TIME", "\t", "LAG TIME", "\t", "EFFICIENCY", "\t", "AOC"
    for well in all_wells:
        
        
        print well, "\t", str(slope_and_lag_dict[well][0]), "\t", str(slope_and_lag_dict[well][1]), "\t", str(efficiency_dict[well]), "\t", str(aoc_dict[well])




if machine == 'TB':  ## for brem lab tecan

   print "\n", 'USING BREM TECAN PROTOCOL FOR FILE:', data_file_name, "\n", 'TOTAL RUN TIME: ', run_time, ' HOURS', "\n"

   timepoints = []
   well_dict = {}
   
   well_nums = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'] 
   rows = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
   

   for row in rows:
      for well_num in well_nums:

         well_dict[row+well_num] = []



   for line in data_file:

      if line[0] == '<':
         continue



      row = line[0]

      line = line.strip().split("\t")[1:]

      for i, read in enumerate(line):
         

         well_dict[row+well_nums[i]].append(float(read))

         
print well_dict

#print well_dict




if machine == 'T':   ## FOR TECAN



 print "\n", 'USING TECAN PROTOCOL FOR FILE:', data_file_name, "\n", 'TOTAL RUN TIME: ', run_time, ' HOURS', "\n"
 



### PUTS ALL THE TIMEPOINTS IN AN ARRAY                                                                                                                                              
 mean_temp = []  # will hold mean temp and std deviation                                                                                                                              
 timepoints = [] # will hold all time_points in minutes                                                                                                                               
 well_dict = {}



 for line in data_file:

    if len(timepoints) == 0:

        line = line.strip().replace("s", "").split("\t")
        for timepoint in line:
            timepoints.append(int(timepoint)/60.0)

        continue



#### GETS THE MEAN AND STD OF ALL TEMPERATURE VALUES ACROSS THE PLATE                                                                                                                


    if len(mean_temp) == 0:
        all_temps = []
        line = line.strip().split("\t")
        for temp in line:

            all_temps.append(float(temp.split(" ")[0]))


        temp_array = np.array(all_temps)

        mean_temp.append(temp_array.mean())
        mean_temp.append(temp_array.std())

        continue

    ### EXITS THE LOOP WHEN ALL WELLS HAVE BEEN READ                                                                                                                                 


    if "Date" in line:

        break

    line = line.strip().split("\t")
## puts a 0 in fron of all the singe numbers to enable sorting later                                                                                                                 
    well = line[-1]
    if len(well) == 2:
        well = list(well)
        well.append(well[1])
        well[1] = "0"
        well = "".join(well)

    well_dict[well] = [float(i) for i in line[:-1]]


for well in well_dict:
   print well_dict[well][:3]




### FOR SPECTRAMAX



if machine == 'S':
 print "\n", 'USING SPECTRAMAX PROTOCOL FOR FILE:', data_file_name, "\n", 'TOTAL RUN TIME: ', run_time, ' HOURS', "\n" 
        
 timepoints = [] # will hold all time_points in minutes
 well_dict = {}
 well_name_list = []

 ### PUTS ALL THE TIMEPOINTS IN AN ARRAY


 data = 'n'

 l = data_file.readline()

 l = l.split("\r")

 #print l

 for line in l:
 #   print line
# for line in data_file:
    

#    print line.split("\t")

    if (len(line.split("\t"))) < 2:
        continue

    if data == 'y' and len(line.strip().split("\t")) < 90:
       data ='n'


    if line.split("\t")[2] == "A1":
#       print 'A1' 
       for well_name in line.strip().split("\t")[2:]:
            if len(well_name) == 2:
                well = well_name[0]+"0"+well_name[1]
            else:
                well = well_name
        
            well_name_list.append(well)

            well_dict[well] = []
    
 #      print len(well_dict)

    if line.split("\t")[1] == '0.00':
        data = 'n'



    if line.split("\t")[0] == '10:00':   ## skips the first timepoint
        data = 'y'

    if data == 'y':
        
       time = line.split("\t")[0]
       if len(time.split(":")) == 2:
           time = "0:" + time

       timepoints.append((float(time.split(":")[0])*60) + float(time.split(":")[1]))
       
       
           
       for pos, well in enumerate(well_name_list):

  #         print len(line.strip().split("\t"))
  #         print line.strip().split("\t")
           well_dict[well].append(float(line.strip().split("\t")[2+pos]))







#print well_dict
#print well_dict['A01']


#print well_dict
 

smooth_dict = Smooth_data(well_dict)

#init_value_dict = Get_init_values(smooth_dict)

#print init_value_dict

aoc_dict = Get_AOC(smooth_dict)

efficiency_dict = Get_efficiency(smooth_dict)

corrected_well_dict = Correct_curves(smooth_dict)

slope_and_lag_dict = Get_double_time_and_lag(corrected_well_dict, machine, run_time)



Print_output(efficiency_dict, slope_and_lag_dict, aoc_dict)
'''

from matplotlib import pyplot as plt 


all_tm = range(0, len(smooth_dict['A01']))


for strain in sorted(in_wells):
  # print_strains[strain] = []                                                                                                                                                                              
#   print strain                                                                                                                                                                                            
#   count+=1  ## fr colors                                                                                                                                                                                   

   print strain
   strain_mean = []
   strain_error = []

   for timepoint in range(len(smooth_dict['A01'])):
      timepointlist = []

      for well in in_wells[strain]:
         timepointlist.append(smooth_dict[well][timepoint])



      strain_mean.append(np.array(timepointlist).mean())
      strain_error.append(np.array(timepointlist).std())

   print len(all_tm), len(strain_mean)



   if 'xxSbay' in strain:
      plt.errorbar(all_tm, strain_mean, label=strain, lw=7, yerr = strain_error, elinewidth=1, color='#FF69B4')

   else:


      plt.errorbar(all_tm, strain_mean, label=strain, lw=7, yerr = strain_error, elinewidth=1)


if legend == 'y':
   
   plt.legend(loc=2)

plt.show()
'''
