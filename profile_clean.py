#/usr/bin/env python
'Removes profiled output of code that never ran'
from __future__ import print_function, division
import sys
import operator

if __name__ == '__main__':
    # Only profiled functions that are run are printed
    input_fname = sys.argv[1]

    with open(input_fname) as file_:
        text = file_.read()
        list_ = text.split('File:')
        for ix in xrange(1, len(list_)):
            list_[ix] = 'File: ' + list_[ix]

        prefix_list = []
        timemap = {}
        for ix in xrange(len(list_)):
            block = list_[ix]
            time_key = 'Total time:'
            timepos = block.find(time_key)
            if timepos == -1:
                prefix_list.append(block)
                continue
            timepos += len(time_key)
            nlpos = block[timepos:].find('\n')
            timestr = block[timepos:timepos + nlpos]
            total_time = float(timestr.replace('s', '').strip())
            if total_time != 0:
                if not total_time in timemap:
                    timemap[total_time] = []
                timemap[total_time].append(block)

        sorted_lists = sorted(timemap.iteritems(), key=operator.itemgetter(0))
        newlist = prefix_list[:]
        for key, val in sorted_lists:
            newlist.extend(val)

        output_text = '\n'.join(newlist)
        if len(sys.argv) > 2:
            output_fname = sys.argv[2]
            with open(output_fname, 'w') as file2_:
                file2_.write(output_text)
        else:
            print(output_text)
            pass
