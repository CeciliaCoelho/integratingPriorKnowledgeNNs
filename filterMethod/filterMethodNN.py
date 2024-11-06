def dominated(filterT, f_w, h_w):
    discard = True
    accept = False
    h_max = 100
    filter_max = 20 

    if len(filterT) > filter_max:
        filterT, h_max = reduceFilter(filterT, h_max)
        print("WARNING: filter reduced. New h_max: ", h_max)

    if h_w >= h_max:
        return filterT, accept

    if filterT == []:
        filterT.append([f_w.item(), h_w.item()])
    else:
            for i in filterT:
                if h_w < i[1] or f_w < i[0]:
                    print("(",f_w.item(),",", h_w.item(),")" ,"accepted by filter")
                    filterT = list(filter(lambda x: x[1] < h_w or x[0] < f_w, filterT))
                    filterT.append([f_w.item(), h_w.item()])
                    discard = False
                    accept = True
                    break

    if discard:
        print("(",f_w.item(),",", h_w.item(),")" ,"not accepted by filter")

    print("filter now: ",filterT)
    return filterT, accept

def reduceFilter(filterT, h_max):

    filterT.sort(key = lambda x: x[1])
    h_max = filterT[19][1]*10

    return filterT[:20], h_max


def filterMethod(filterT, w_k, f_w, h_w):

        filterT, accept = dominated(filterT, f_w, h_w)
        return filterT, accept


