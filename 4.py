Text = "Dude!!!! And I thought I knew a lotttt. Phewwwww!\
I won’t back down. At least I understand now Daaata Science \
is much more than what we are taught in MOOOCs. That is allllright. \
I won’t get demotivated. I’ll work harder and in noooo time, I’ll \
get better & be backkk next time."

result = [Text[0], Text[1]]
for i in range(2, len(Text)):
    if Text[i] == Text[i-1] and Text[i] == Text[i-2]:
        continue
    result.append(Text[i])
result = ''.join(result)
print(result)


#complexity analysis line by line:
#7     > list initialization: Constant complexity, O(1)
#8-11  > loop N = Length of string "Text" - 2 times:
#        9,11 contains conditional check performed in constant complecity, O(1)
#        11 contains list append operation which is also constant time operation, O(1)
#        So, 8-11 has a complexity of O(1) * (N-2) = O(N-2) approximately, O(N)
#12    > joining N element list into a string performed in linear time O(N)
#13    > printing N element string in linear time, O(N)
#--------------------------------------------------------------------------------------
#TOTAL = O(1) + O(N) + O(N) + O(N) = O(1) + 3*O(N)
#Ignoring contants, Approximately, O(N) 
