
lables = set( ['B-alarm/alarm_modifier', 'I-reminder/reference', 'B-reminder/reminder_modifier', 'I-reminder/todo', 'NoLabel',
          'B-timer/attributes', 'B-datetime', 'B-reminder/todo', 'B-reminder/recurring_period', 'B-timer/noun', 'I-weather/noun',
          'B-negation', 'B-reminder/noun', 'I-weather/attribute', 'I-alarm/alarm_modifier', 'B-weather/noun', 'I-datetime', 'B-weather/attribute',
          'I-reminder/recurring_period', 'I-location', 'B-demonstrative_reference', 'B-location', 'I-reminder/reminder_modifier', 'B-reminder/reference',
          'B-weather/temperatureUnit', 'I-reminder/noun', 'B-news/type', 'I-demonstrative_reference', 'I-negation', 'B-alarm/recurring_period', "I-alarm/recurring_period"])
lab2 = set()



with open("/home/santi/BA/multilingual_task_oriented_dialog_slotfilling/es/train-es.conllu") as f:
    for line in f:
        if line[0] != "#":

           l = line.split()

           if len(l) > 0:
               #print(l[1]," ",l[-1])
               if l[-1] not in lables:
                   print(l[-1])
           else:
               pass

