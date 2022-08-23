def line(input, start, end, type):
    '''
    Returns a straight line from the start to the end
    any value outside the region returns 0
    '''
    if not start <= input <= end:
        return 0

    if type == 'inc':
        slope = 1/(end-start)
        output = slope * (input - start)
        return (output + abs(output))/2
    elif type == 'dec':
        slope = -1/(end-start)
        output =  slope * (input - start) + 1
        return (output + abs(output))/2
    elif type == 'flat':
        return 1
    else:
        raise Exception("Unexpected line type")

def get_score(exps, predictions, extra_edges):
    '''
    To be modified to meet the criteria
    '''
    score = 0.0
    cost = extra_edges * 0.5
    score -= cost

    counts = [0]*14
    for fix in exps:
        for result in exps[fix]:
            node = result[0]
            predict_value = predictions[fix][node]
            # experiment showed (ON)
            if result[1] == '1' and result[2] == '0':
                counts[0] += 2
                add = 2*line(predict_value,0.5,1,'inc')
                counts[1] += add
            # experiment showed (ON/some ON)
            elif result[1] == '1' and result[2] == '1':
                counts[2] += 1
                add = max(line(predict_value,0.3,0.5,'inc'),line(predict_value,0.5,1,'flat'))
                counts[3] += add
            # experiment showed (some ON)
            elif result[1] == '0' and result[2] == '1' and result[3] == '0':
                # find the wildtype
                for fix1 in fix:
                    if fix1[0] == 'ABA':
                        value = fix1[1]
                        break
                wt = (('ABA', value),)
                wt_behaviors = exps[wt]
                # print('WT behavior:', wt_behaviors)
                wt_behavior = None
                for wtb1 in wt_behaviors:
                    if wtb1[0] == node:
                        wt_behavior = wtb1

                if wt_behavior == None or wt_behavior[2] == '1':
                    counts[4] += 1
                    add = max(line(predict_value,0.3,0.5,'inc'),line(predict_value,0.5,0.7,'dec'))
                    counts[5] += add
                elif wt_behavior[3] == '1':
                    counts[6] += 1
                    add = max(line(predict_value,0.3,0.5,'inc'),line(predict_value,0.5,1,'dec'))
                    counts[7] += add
                else: # wt_behavior[1] == '1'
                    counts[8] += 1
                    add = max(line(predict_value,0,0.5,'inc'),line(predict_value,0.5,0.7,'dec'))
                    counts[9] += add
            # experiment showed (some ON/OFF)
            elif result[2] == '1' and result[3] == '1':
                counts[10] += 1
                add = max(line(predict_value,0,0.5,'flat'),line(predict_value,0.5,0.7,'dec'))
                counts[11] += add
            # experiment showed (OFF)
            elif result[2] == '0' and result[3] == '1':
                counts[12] +=1
                add = line(predict_value,0,0.5,'dec')
                counts[13] += add
            else:
                raise Exception("Unexpected experiment input")

            score += add
            print("- - - - - - - - - -")
            print("Perturbation: ", fix)
            print("Experiment: ", result)
            print("Prediction: ", predict_value)
            print("adding ", add)
            print("Current score ", score)
    print()
    print("- - - - - - - - - -")
    print("total score: ",score, '/', counts[0]+counts[2]+counts[4]+counts[6]+counts[8]+counts[10]+counts[12])
    print("edge cost: ", cost)
    print("score got from ON: ", counts[1], '/', counts[0])
    print("score got from ON/some: ", counts[3], '/', counts[2])
    print("score got from some: ", counts[5], '/', counts[4])
    print("score got from increased: ", counts[7], '/', counts[6])
    print("score got from reduced: ", counts[9], '/', counts[8])
    print("score got from some/OFF: ", counts[11], '/', counts[10])
    print("score got from OFF: ", counts[13], '/', counts[12])
    return score

def get_score_new(exps, predictions, extra_edges):
    '''
    To be modified to meet the criteria
    '''
    total = 0.0
    cost = extra_edges * 0.5
    total -= cost

    for expset in exps:
        score = float(expset[0])
        exp = expset[1]
        for fix in exp:
            result = exp[fix]
            node = result[0]
            predict_value = predictions[fix][node]
            # experiment showed (ON)
            if '1' in result and '0.5' not in result:
                mult = line(predict_value,0.5,1,'inc')
            # experiment showed (ON/some ON)
            elif '1' in result and '0.5' in result:
                mult = max(line(predict_value,0.3,0.5,'inc'),line(predict_value,0.5,1,'flat'))
            # experiment showed (some ON)
            elif '0.5' in result and '0' not in result:
                mult = max(line(predict_value,0.3,0.5,'inc'),line(predict_value,0.5,0.7,'dec'))
            # experiment showed (some ON/OFF)
            elif '0.5' in result and '0' in result:
                mult = max(line(predict_value,0,0.5,'flat'),line(predict_value,0.5,0.7,'dec'))
            # experiment showed (OFF)
            elif '0.5' not in result and '0' in result:
                mult = line(predict_value,0,0.5,'dec')
            else:
                raise Exception("Unexpected experiment input")
            score *= mult

            print("- - - - - - - - - -")
            print("Perturbation: ", fix)
            print("Experiment: ", result)
            print("Prediction: ", predict_value)
            print("multiplying ", mult)
            print("Current score ", score)
        total += score

    return total
