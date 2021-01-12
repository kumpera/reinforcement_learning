#! /usr/bin/env python3 -W ignore::DeprecationWarning
from reinforcement_learning.messages.flatbuff.v2.EventBatch import *
from reinforcement_learning.messages.flatbuff.v2.BatchMetadata import *
from reinforcement_learning.messages.flatbuff.v2.EventBatch import *
from reinforcement_learning.messages.flatbuff.v2.LearningModeType import LearningModeType
from reinforcement_learning.messages.flatbuff.v2.PayloadType import PayloadType
from reinforcement_learning.messages.flatbuff.v2.OutcomeValue import OutcomeValue
from reinforcement_learning.messages.flatbuff.v2.NumericOutcome import NumericOutcome
from reinforcement_learning.messages.flatbuff.v2.NumericIndex import NumericIndex

from reinforcement_learning.messages.flatbuff.v2.CbEvent import CbEvent
from reinforcement_learning.messages.flatbuff.v2.OutcomeEvent import OutcomeEvent
from reinforcement_learning.messages.flatbuff.v2.MultiSlotEvent import MultiSlotEvent
from reinforcement_learning.messages.flatbuff.v2.CaEvent import CaEvent
from reinforcement_learning.messages.flatbuff.v2.DedupInfo import DedupInfo

from reinforcement_learning.messages.flatbuff.v2.KeyValue import *  
from reinforcement_learning.messages.flatbuff.v2.TimeStamp import *  
from reinforcement_learning.messages.flatbuff.v2.FileHeader import *  
from reinforcement_learning.messages.flatbuff.v2.SerializedBatch import *  
from reinforcement_learning.messages.flatbuff.v2.JoinedPayload import *  
from reinforcement_learning.messages.flatbuff.v2.BatchType import *  
from reinforcement_learning.messages.flatbuff.v2.Metadata import *  
from reinforcement_learning.messages.flatbuff.v2.Event import *  


import flatbuffers
import zstandard as zstd
import sys
import json
import struct
from datetime import datetime
import numpy as np

"""
TODO:
Incremental join instead of loading all interactions at once
Respect EUD.
"""

class PreambleStreamReader:
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
            # yield (EventBatch.GetRootAsEventBatch(msg, 0), msg)
            yield msg


def mk_weirdtime(builder, time=None, weirdTime=None):
    if weirdTime != None:
        return CreateTimeStamp(builder, weirdTime.Year(), weirdTime.Month(), weirdTime.Day(), weirdTime.Hour(), weirdTime.Minute(), weirdTime.Second(), weirdTime.Subsecond())

    if time == None:
        time = datetime.utcnow()
    return CreateTimeStamp(builder, time.year, time.month, time.day, time.hour, time.minute, time.second, int(time.microsecond))

def mk_serialized_batch(builder, kind, payload):
    off = mk_bytes_vector(builder, payload)

    SerializedBatchStart(builder)
    SerializedBatchAddType(builder, kind)
    SerializedBatchAddPayload(builder, off)

    return SerializedBatchEnd(builder)

def create_string_or_none(builder, string):
    if string != None:
        return builder.CreateString(str(string))
    return None

def reserialize_event(builder, evt):
    md_off = None
    meta = evt.Meta()
    if meta != None:
        evt_id = create_string_or_none(builder, meta.Id())
        app_id = create_string_or_none(builder, meta.AppId())

        MetadataStart(builder)
        if evt_id:
            MetadataAddId(builder, evt_id)
        MetadataAddClientTimeUtc(builder, mk_weirdtime(builder, weirdTime=meta.ClientTimeUtc()))
        if app_id:
            MetadataAddAppId(builder, app_id)
        MetadataAddPayloadType(builder, meta.PayloadType())
        MetadataAddPassProbability(builder, meta.PassProbability())
        md_off = MetadataEnd(builder)

    payload_off = mk_bytes_vector(builder, evt.PayloadAsNumpy())

    EventStart(builder)
    EventAddMeta(builder, md_off)
    EventAddPayload(builder, payload_off)
    return EventEnd(builder)

def mk_offsets_vector(builder, arr, startFun):
    startFun(builder, len(arr))
    for i in reversed(range(len(arr))):
        builder.PrependUOffsetTRelative(arr[i])
    return builder.EndVector(len(arr))

def mk_bytes_vector(builder, arr):
    return builder.CreateNumpyVector(np.array(list(arr), dtype='b'))

MSG_TYPE_HEADER = 0x55555555
MSG_TYPE_REGULAR = 0xFFFFFFFF
MSG_TYPE_EOF = 0xAAAAAAAA

class BinLogWriter:
    def __init__(self, file_name):
        self.file = open(file_name, 'wb')
    
    def write_message(self, kind, payload):
        padding_bytes = len(payload) % 8
        print(f'msg {kind:X} size: {len(payload)} padding {padding_bytes}')

        self.file.write(struct.pack('I', kind))
        self.file.write(struct.pack('I', len(payload)))
        self.file.write(payload)
        if padding_bytes > 0:
            self.file.write(bytes([0] * padding_bytes))

    def write_header(self, properties):
        self.file.write(b'VWFB')
        self.file.write(struct.pack('I', 1))

        builder = flatbuffers.Builder(0)
        kv_offsets = []
        for key in properties:
            value = properties[key]
            k_off = builder.CreateString(str(key))
            v_off = builder.CreateString(str(value))
            KeyValueStart(builder)
            KeyValueAddKey(builder, k_off)
            KeyValueAddValue(builder, v_off)
            kv_offsets.append(KeyValueEnd(builder))

        props_off = mk_offsets_vector(builder, kv_offsets, FileHeaderStartPropertiesVector)

        FileHeaderStart(builder)
        FileHeaderAddJoinTime(builder, mk_weirdtime(builder))
        FileHeaderAddProperties(builder, props_off)

        header_off = FileHeaderEnd(builder)
        builder.Finish(header_off)
        self.write_message(MSG_TYPE_HEADER, builder.Output())

    def write_join_msg(self, iteractions_bytes, obs):
        builder = flatbuffers.Builder(0)

        iter_off = mk_serialized_batch(builder, BatchType.Interactions, iteractions_bytes)

        #create an observation batch
        b2 = flatbuffers.Builder(0)
        evt_offs = []
        for evt_lst in obs:
            for evt in evt_lst:
                evt_offs.append(reserialize_event(b2, evt))

        evts_off = mk_offsets_vector(b2, evt_offs, EventBatchStartEventsVector)

        content_encoding_off = b2.CreateString("IDENTITY")
        BatchMetadataStart(b2)
        BatchMetadataAddContentEncoding(b2, content_encoding_off)
        md_off = BatchMetadataEnd(b2)
        
        EventBatchStart(b2)
        #TODO batch metadata
        EventBatchAddMetadata(b2, md_off)
        EventBatchAddEvents(b2, evts_off)
        b2.Finish(EventBatchEnd(b2))

        obs_off = mk_serialized_batch(builder, BatchType.Observations, b2.Output())

        batches_off = mk_offsets_vector(builder, [iter_off, obs_off], JoinedPayloadStartBatchesVector)

        JoinedPayloadStart(builder)
        JoinedPayloadAddBatches(builder, batches_off)
        builder.Finish(JoinedPayloadEnd(builder))

        self.write_message(MSG_TYPE_REGULAR, builder.Output())

    def write_eof(self):
        self.write_message(MSG_TYPE_EOF, b'')


# interactions_file = PreambleStreamReader('iter.fb')
# observations_file = PreambleStreamReader('obs.fb')
interactions_file = PreambleStreamReader('large_it.fb')
observations_file = PreambleStreamReader('large_obs.fb')
# in_f = open(interactions_file, "rb")
result_file = 'merged.log'

observations = dict()
obs_count = 0
obs_ids = 0
for msg in observations_file.messages():
    batch = EventBatch.GetRootAsEventBatch(msg, 0)
    for i in range(0, batch.EventsLength()):
        evt = batch.Events(i)
        # print(f'got evt {evt.Meta().Id()} kind:{evt.Meta().PayloadType()}')
        evt_id = evt.Meta().Id()
        if evt_id not in observations:
            observations[evt_id] = []
            obs_ids += 1
        observations[evt_id].append(evt)
        obs_count +=1

print(f'found {obs_count} observations with {obs_ids} ids')
# join_streams(interactions_file observations_file, result_file)
bin_f = BinLogWriter(result_file)
bin_f.write_header({ 'eud': '-1', 'joiner': 'joiner.py', 'reward': 'latest'})

for msg in interactions_file.messages():
    batch = EventBatch.GetRootAsEventBatch(msg, 0)
    #collect all observations
    used_obs = 0
    obs = []
    for i in range(0, batch.EventsLength()):
        evt = batch.Events(i)
        evt_id = evt.Meta().Id()
        if evt_id in observations:
            obs.append(observations[evt_id])
            used_obs += len(observations[evt_id])
    print(f'batch with {len(obs)} iteractions and {used_obs} observations')
    print(f'joining iters with {batch.Metadata().ContentEncoding()} len:{len(msg)}')
    bin_f.write_join_msg(msg, obs)

bin_f.write_eof()