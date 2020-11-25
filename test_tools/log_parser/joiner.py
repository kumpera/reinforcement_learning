#! /usr/bin/env python3 -W ignore::DeprecationWarning
import reinforcement_learning.messages.flatbuff.v2.CbEvent as cb
import reinforcement_learning.messages.flatbuff.v2.EventBatch as eb
import reinforcement_learning.messages.flatbuff.v2.DedupInfo as di
import reinforcement_learning.messages.flatbuff.v2.PayloadType as pt
import reinforcement_learning.messages.flatbuff.v2.LearningModeType as lm
import flatbuffers
import zstandard as zstd
import sys
import json

# PREAMBLE_LENGTH = 8

# def parse_preamble(buf):
#     reserved = buf[0]
#     version = buf[1]
#     msg_type = int.from_bytes(buf[2:4], "big")
#     msg_size = int.from_bytes(buf[4:8], "big")
#     return { 'reserved': reserved, 'version': version, 'msg_type': msg_type, 'msg_size': msg_size}

# def payload_name(payload):
#     for k in [f for f in dir(pt.PayloadType) if not f.startswith('__')]:
#         if getattr(pt.PayloadType, k) == payload:
#             return k
#     return f'<unk_{payload}>'

# def learning_mode_name(learning_mode):
#     for k in [f for f in dir(lm.LearningModeType) if not f.startswith('__')]:
#         if getattr(lm.LearningModeType, k) == learning_mode:
#             return k
#     return f'<unk_{learning_mode}>'


# def process_payload(payload, is_dedup):
#     if not is_dedup:
#         return payload
#     return zstd.decompress(payload)

# def parse_cb(payload):
#     evt = cb.CbEvent.GetRootAsCbEvent(payload, 0)
#     print(f'\tcb: actions:{evt.ActionIdsLength()} ctx:{evt.ContextLength()} probs:{evt.ProbabilitiesLength()} lm:{learning_mode_name(evt.LearningMode())}')
#     payload = json.loads(bytearray(evt.ContextAsNumpy()).decode('utf-8'))
#     print('\tcontext:')
#     print(json.dumps(payload, indent = 1))

# def parse_action_dict(payload):
#     evt = di.DedupInfo.GetRootAsDedupInfo(payload, 0)
#     print(f'\t\tad: ids:{evt.IdsLength()} values:{evt.ValuesLength()}')
#     for i in range(0, evt.ValuesLength()):
#         print(f'\t\t\t[{evt.Ids(i)}]: "{evt.Values(i)}"')

# def dump_event(evt, idx, is_dedup):
#     m = evt.Meta()
#     print(f'\t[{idx}] id:{m.Id()} type:{payload_name(m.PayloadType())} payload-size:{evt.PayloadLength()}')

#     payload = process_payload(evt.PayloadAsNumpy(), is_dedup)

#     if m.PayloadType() == pt.PayloadType.DedupInfo:
#         parse_action_dict(payload)
#     elif m.PayloadType() == pt.PayloadType.CB:
#         parse_cb(payload)


# def dump_file(f):
#     buf = open(f, 'rb').read()
#     buf = bytearray(buf)
#     preamble = parse_preamble(buf)

#     batch = eb.EventBatch.GetRootAsEventBatch(buf[PREAMBLE_LENGTH : PREAMBLE_LENGTH + preamble["msg_size"]], 0)
#     meta = batch.Metadata()
#     print(f'parsed {f} with {batch.EventsLength()} events preamble:{preamble} enc:{meta.ContentEncoding()}')
#     is_dedup = b'ZSTD_AND_DEDUP' == meta.ContentEncoding()
#     for i in range(0, batch.EventsLength()):
#         dump_event(batch.Events(i), i, is_dedup)
#     print("----\n")

# for f in sys.argv[1:]:
#     dump_file(f)

class PreambleStream:
    def __init__(self, file_name):
        self.file = open(file_name, 'rb')
    
    def parse_preamble(self):
        buf = self.file.read(8)
        if buf == b'':
            return None

        reserved = buf[0]
        version = buf[1]
        msg_type = int.from_bytes(buf[2:4], "big")
        msg_size = int.from_bytes(buf[4:8], "big")
        return { 'reserved': reserved, 'version': version, 'msg_type': msg_type, 'msg_size': msg_size}

    def messages(self):
        while True:
            header = self.parse_preamble()
            if header == None:
                break
            msg = self.file.read(header['msg_size'])
            yield eb.EventBatch.GetRootAsEventBatch(msg, 0)


"""
TODO:
Incremental join instead of loading all interactions at once
Respect EUD.
"""
interactions_file = PreambleStream('iter.fb')
observations_file = PreambleStream('obs.fb')
# in_f = open(interactions_file, "rb")
result_file = 'merged.fb'

observations = dict()
for batch in observations_file.messages():
    for i in range(0, batch.EventsLength()):
        evt = batch.Events(i)
        # print(f'got evt {evt.Meta().Id()} kind:{evt.Meta().PayloadType()}')
        evt_id = evt.Meta().Id()
        if evt_id not in observations:
            observations[evt_id] = []
        observations[evt_id].append(evt)

for batch in interactions_file.messages():
    for i in range(0, batch.EventsLength()):
        evt = batch.Events(i)
        evt_id = evt.Meta().Id()
        print(f'evt {evt_id} has_reward:{evt_id in observations}')

# join_streams(interactions_file observations_file, result_file)
