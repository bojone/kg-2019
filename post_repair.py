#! -*- coding: utf-8 -*-

from __future__ import print_function
import json
import uniout
import re


f = open('test_pred.json')
F = open('test_pred2.json', 'w')

orders = ['subject', 'predicate', 'object', 'object_type', 'subject_type']
company_names = [
    u'集团', u'公司', u'有限公司', u'电视台', u'集团公司', u'文化传媒有限公司', u'传媒有限公司', u'股份有限公司',
    u'有限责任公司', u'工作室', u'株式会社', u'中心', u'影视传媒', u'传媒股份有限公司', u'保险有限公司'
]

n = 0
for t in f:
    t = json.loads(t)
    spos = []
    for spo in t['spo_list']:
        if spo['predicate'] == u'目' and spo['subject'][-1] in [u'科', u'属']:
            # print t, spo
            n += 1
            continue
        elif spo['predicate'] == u'国籍' and spo[
                'object'] == u'中国' and u'中国共产党' in t['text'] and t[
                    'text'].count(u'中国') == 1:
            # print t, spo
            n += 1
            continue
        elif spo['predicate'] == u'作者' and (spo[
                'object'][-2:] in [u'公司', u'学会'] or spo['object'][-3:] in [u'出版社', u'委员会']):
            # print t, spo
            n += 1
            continue
        elif spo['object'] == spo['subject'] and spo['predicate'] in [
                u'父亲', u'母亲', u'丈夫', u'妻子']:
            # print t, spo
            n += 1
            continue
        elif spo['predicate'] == u'气候' and len(
                spo['object']) >= 2 and t['text'][t['text'].find(spo['object'])
                                                  - 1] in [
                                                      u'南', u'北', u'亚', u'中'
                                                  ]:
            # print t, spo
            n += 1
            spo['object'] = t['text'][t['text'].find(spo['object']) -
                                      1] + spo['object']
            spos.append(spo)
        elif spo['predicate'] == u'祖籍' and spo['object'][-1] == u'人':
            # print t, spo
            n += 1
            spo['object'] = spo['object'][:-1]
            spos.append(spo)
        elif spo['predicate'] == u'出版社' and spo['object'][
                -3:] != u'出版社' and spo['object'] + u'出版社' in t['text']:
            # print t, spo
            n += 1
            spo['object'] = spo['object'] + u'出版社'
            spos.append(spo)
        elif spo['predicate'] in [
                u'成立日期', u'注册资本', u'董事长', u'创始人', u'总部地点', u'占地面积', u'简称'
        ]:
            whocare = 0
            for com in company_names:
                if spo['subject'][-len(com):] != com and spo[
                        'subject'] + com in t['text']:
                    # print t, spo
                    n += 1
                    spo['subject'] = spo['subject'] + com
                    spos.append(spo)
                    whocare = 1
            if whocare == 0:
                spos.append(spo)
        elif spo['predicate'] == u'出品公司':
            whocare = 0
            for com in company_names:
                if spo['object'][-len(com):] != com and spo[
                        'object'] + com in t['text']:
                    # print t, spo
                    n += 1
                    spo['object'] = spo['object'] + com
                    spos.append(spo)
                    whocare = 1
            if whocare == 0:
                spos.append(spo)
        elif re.findall(u'[、，]', spo['object']):
            # print t, spo
            n += 1
            for o in re.split(u'[、，]', spo['object']):
                _ = spo.copy()
                _['object'] = o
                spos.append(_)
                # print _
        else:
            spos.append(spo)
    if not spos:
        for spo in t['spo_list']:
            if spo['predicate'] == u'目' and spo['subject'][-1] == u'属':
                spos.append(spo)
                break
    if not spos:
        for spo in t['spo_list']:
            if spo['predicate'] == u'目' and spo['subject'][-1] == u'科':
                spos.append(spo)
                break
    R_dic = {}
    for spo in spos:
        spo = (spo['subject'], spo['predicate'], spo['object'])
        if spo[:2] not in R_dic:
            R_dic[spo[:2]] = set()
        R_dic[spo[:2]].add(spo[2])
    R = set()
    for k,v in R_dic.items():
        v = list(v)
        if len(v) == 2 and v[0].encode('utf-8').isalpha() and v[1].encode('utf-8').isalpha():
            if '%s %s' % (v[0], v[1]) in t['text'].lower():
                # print t
                n += 1
                R.add((k[0], k[1], '%s %s' % (v[0], v[1])))
            elif '%s %s' % (v[1], v[0]) in t['text'].lower():
                # print t
                n += 1
                R.add((k[0], k[1], '%s %s' % (v[1], v[0])))
            else:
                for o in v:
                    R.add((k[0], k[1], o))
        elif len(v) == 2 and not re.findall(u'[^\u4e00-\u9fa5]', v[0]) and not re.findall(u'[^\u4e00-\u9fa5]', v[1]):
            whocare = False
            if v[1] in v[0]:
                a, b = v[1], v[0]
                i = b.find(a)
                if i == 0:
                    if len(b) > i + len(a):
                        if b[i + len(a)] not in [u'和', u'的']:
                            whocare = True
                    else:
                        whocare = True
                elif i > 0 and b[i-1] not in [u'和', u'的']:
                    if len(b) > i + len(a):
                        if b[i + len(a)] not in [u'和', u'的']:
                            whocare = True
                    else:
                        whocare = True
            elif v[0] in v[1]:
                a, b = v[0], v[1]
                i = b.find(a)
                if i == 0:
                    if len(b) > i + len(a):
                        if b[i + len(a)] not in [u'和', u'的']:
                            whocare = True
                    else:
                        whocare = True
                elif i > 0 and b[i-1] not in [u'和', u'的']:
                    if len(b) > i + len(a):
                        if b[i + len(a)] not in [u'和', u'的']:
                            whocare = True
                    else:
                        whocare = True
            if whocare:
                # print t
                # print k, a, b
                n += 1
                R.add((k[0], k[1], b))
            else:
                for o in v:
                    R.add((k[0], k[1], o))
        else:
            for o in v:
                R.add((k[0], k[1], o))
    R_dic = {}
    for spo in R:
        spo = spo[::-1]
        if spo[:2] not in R_dic:
            R_dic[spo[:2]] = set()
        R_dic[spo[:2]].add(spo[2])
    R = set()
    for k,v in R_dic.items():
        v = list(v)
        if len(v) == 2 and not re.findall(u'[^\u4e00-\u9fa5]', v[0]) and not re.findall(u'[^\u4e00-\u9fa5]', v[1]):
            whocare = False
            if v[1] in v[0]:
                a, b = v[1], v[0]
                i = b.find(a)
                if i == 0:
                    if len(b) > i + len(a):
                        if b[i + len(a)] not in [u'和', u'的']:
                            whocare = True
                    else:
                        whocare = True
                elif i > 0 and b[i-1] not in [u'和', u'的']:
                    if len(b) > i + len(a):
                        if b[i + len(a)] not in [u'和', u'的']:
                            whocare = True
                    else:
                        whocare = True
            elif v[0] in v[1]:
                a, b = v[0], v[1]
                i = b.find(a)
                if i == 0:
                    if len(b) > i + len(a):
                        if b[i + len(a)] not in [u'和', u'的']:
                            whocare = True
                    else:
                        whocare = True
                elif i > 0 and b[i-1] not in [u'和', u'的']:
                    if len(b) > i + len(a):
                        if b[i + len(a)] not in [u'和', u'的']:
                            whocare = True
                    else:
                        whocare = True
            if whocare:
                # print t
                # print k, a, b
                n += 1
                R.add((k[0], k[1], b))
            else:
                for o in v:
                    R.add((k[0], k[1], o))
        else:
            for o in v:
                R.add((k[0], k[1], o))
    R = [spo[::-1] for spo in R]
    R_dic = {}
    for spo in R:
        if spo[0] not in R_dic:
            R_dic[spo[0]] = {}
        R_dic[spo[0]][spo[1]] = spo[2]
    for k, v in R_dic.items():
        if set(v.keys()) == set([u'父亲', u'母亲']):
            a, b = v[u'父亲'], v[u'母亲']
            if (a, u'妻子', b) not in R:
                # print t
                # print (a, u'妻子', b)
                R.append((a, u'妻子', b))
                n += 1
            if (b, u'丈夫', a) not in R:
                # print t
                # print (b, u'丈夫', a)
                R.append((b, u'丈夫', a))
                n += 1
        if u'父亲' in v and u'母亲' not in v:
            if u'妻子' in R_dic.get(v[u'父亲'], {}):
                if k != R_dic[v[u'父亲']][u'妻子']:
                    R.append((k, u'母亲', R_dic[v[u'父亲']][u'妻子']))
                    # print t
                    # print (k, u'母亲', R_dic[v[u'父亲']][u'妻子'])
                    n += 1
        elif u'母亲' in v and u'父亲' not in v:
            if u'丈夫' in R_dic.get(v[u'母亲'], {}):
                if k != R_dic[v[u'母亲']][u'丈夫']:
                    R.append((k, u'父亲', R_dic[v[u'母亲']][u'丈夫']))
                    # print t
                    # print (k, u'父亲', R_dic[v[u'母亲']][u'丈夫'])
                    n += 1
    t['spo_list'] = [dict(zip(orders, spo + ('', ''))) for spo in R]
    s = json.dumps(t, ensure_ascii=False)
    F.write(s.encode('utf-8') + '\n')

print(n)
