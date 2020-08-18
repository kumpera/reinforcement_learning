#pragma once

#pragma once

#include <vector>

#include <flatbuffers/flatbuffers.h>

#include "action_flags.h"
#include "ranking_event.h"
#include "data_buffer.h"
#include "logger/message_type.h"
#include "api_status.h"
#include "utility/data_buffer_streambuf.h"
#include "learning_mode.h"

#include "generated/v2/OutcomeSingle_generated.h"
#include "generated/v2/CbEvent_generated.h"
#include "generated/v2/CcbEvent_generated.h"
#include "generated/v2/SlatesEvent_generated.h"

namespace reinforcement_learning {
  namespace logger {
    messages::flatbuff::v2::LearningModeType GetLearningMode(learning_mode mode);

    template<payload_type pt>
    struct payload_serializer {
      const payload_type type = pt;
    };

    struct cb_serializer : payload_serializer<payload_type::CB> {
      static generic_event::payload_buffer_t&& event(const char* context, unsigned int flags, learning_mode learning_mode, const ranking_response& response) {
        flatbuffers::FlatBufferBuilder fbb;
        std::vector<uint64_t> action_ids;
        std::vector<float> probabilities;
        for (auto const& r : response) {
          action_ids.push_back(r.action_id + 1);
          probabilities.push_back(r.probability);
        }
        std::vector<unsigned char> _context;
        std::string context_str(context);
        copy(context_str.begin(), context_str.end(), std::back_inserter(_context));

        auto fb = messages::flatbuff::v2::CreateCbEventDirect(fbb, flags | action_flags::DEFERRED, &action_ids, &_context, &probabilities, response.get_model_id(), GetLearningMode(learning_mode));
        fbb.FinishSizePrefixed(fb);
        return std::move(fbb.Release());
      }
    };

    struct ccb_serializer : payload_serializer<payload_type::CCB> {
      static generic_event::payload_buffer_t&& event(const char* context, unsigned int flags, const std::vector<std::vector<uint32_t>>& action_ids,
        const std::vector<std::vector<float>>& pdfs, const std::string& model_version) {
        generic_event::payload_buffer_t buf;
        return std::move(buf);
      }
    };

    struct slates_serializer : payload_serializer<payload_type::SLATES> {
      static generic_event::payload_buffer_t&& event(const char* context, unsigned int flags, const std::vector<std::vector<uint32_t>>& action_ids,
        const std::vector<std::vector<float>>& pdfs, const std::string& model_version) {
        generic_event::payload_buffer_t buf;
        return std::move(buf);
      }
    };

    struct outcome_single_serializer : payload_serializer<payload_type::OUTCOME_SINGLE> {
      static generic_event::payload_buffer_t&& event(float outcome) {
        generic_event::payload_buffer_t buf;
        return std::move(buf);
      }

      static generic_event::payload_buffer_t&& event(const char* outcome) {
        generic_event::payload_buffer_t buf;
        return std::move(buf);
      }

      static generic_event::payload_buffer_t&& report_action_taken() {
        generic_event::payload_buffer_t buf;
        return std::move(buf);
      }
    };
  }
}