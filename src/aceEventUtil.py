

# event = [sentence_ldc_scope, eventType, eventSubType, anchor, (arg, role), (arg, role), ...]
# eventString: sentence[sep]eventType[sep]eventSubType[sep]anchor[sep]arg[sep]role[sep]arg[sep]role......
def str2event(eventString, separator):
    arr = eventString.split(separator)
    if len(arr)%2 != 0:
        print "--Error! #eventItem", len(arr)
        print "--debug eventString:", eventString
        print "--debug event arr:", arr
    arguments = arr[4:]
    arguments = [(arguments[i*2], arguments[i*2+1]) for i in range(len(arguments)/2)]

    event = arr[:4]
    event.extend(arguments)
    return event


# event = [sentence_ldc_scope, eventType, eventSubType, anchor, (arg, role), (arg, role), ...]
# output: sentence[sep]eventType[sep]eventSubtype[sep]anchor[sep]arg[sep]role[sep]arg[sep]role......
def event2str(event, separator):
    arguments = [arg[0]+separator+arg[1] for arg in event[4:]]
    newArrangedEvent = event[:4]
    newArrangedEvent.extend(arguments)
    eventString = separator.join(newArrangedEvent)
    #print eventString
    return eventString

def argTrain2str(argTrain, sepFst, sepSnd):
    sentId_str, sentence, eventType, idxs, role = argTrain
    idxs_str = str(idxs[0]) + sepSnd + str(idxs[1])
    return sepFst.join([sentId_str, sentence, idxs_str, role, eventType])

def str2ArgTrain(string, sepFst, sepSnd):
    (sentence, idxs_str, role) = string.split(sepFst)
    idxs = idxs_str.split(sepSnd)
    idxs = [int(item) for item in idxs]
    return (sentence, idxs, role)

# output: sentence[sepFst]arg1[sepSnd]role1[sepSnd]arg2[sepSnd]role2[sepSnd]arg3[sepSnd]role3[sepFst]arg1[sepSnd]arg2
def event2TrainFormatStr_noType(event, sepFst, sepSnd, Arguments):
    roles_Pos = [arg[1] for arg in event[4:]]
    arguments_Pos = sepSnd.join([arg[0]+sepSnd+arg[1] for arg in event[4:]])
    # add trigger as extra argument
    arguments_Pos = event[3]+sepSnd+"trigger"+sepSnd+arguments_Pos
    # negtive argument instance
    roles_Neg = list(Arguments-set(roles_Pos))
    arguments_Neg = sepSnd.join(roles_Neg)
    return sepFst.join([event[0], arguments_Pos, arguments_Neg])

# event: [sentence, positive arg-role pair list, negtive role list]
def trainFormatStr2event_noType(eventStr, sepFst, sepSnd):
    sentence, argStr_Pos, argStr_Neg = eventStr.split(sepFst)
    roles_Neg = argStr_Neg.split(sepSnd)
    arg_roles_Pos = argStr_Pos.split(sepSnd)
    arg_roles_Pos = [(arg_roles_Pos[2*i], arg_roles_Pos[2*i+1]) for i in range(len(arg_roles_Pos)/2)]
    return [sentence, arg_roles_Pos, roles_Neg]

def loadEventHierarchy(filename):
    eventSubTypeRoleHash = {}
    eventSubTypeHash = {}
    content = open(filename, "r").readlines()
    eventStructures = "".join(content).lower().split("\n\n")[1:]
    for eventStructure in eventStructures:
        if len(eventStructure.strip())==0: continue
        arr = eventStructure.strip().split("\n")
        eventType = arr[0]
        #print eventType
        for line in arr[1:]:
            eventSubType = line[:line.find(":")]
            argRoles = line[line.find(":")+1:].split()
            #print (eventSubType, argRoles)
            eventSubTypeRoleHash[eventSubType] = argRoles
            eventSubTypeHash[eventSubType] = eventType
    #print len(eventSubTypeHash)
    return eventSubTypeHash, eventSubTypeRoleHash


def outputEventFormat2(event):
    sentence, subtype, arg_role_Pos, role_Neg = event
    print "----Event:"
    print "-- sent:", sentence
    print "-- subtype:", subtype
    print "-- arg-role-pos:", arg_role_Pos
    print "-- role-neg:", role_Neg
