/*
 Copyright 2016-2020 Intel Corporation
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 
     http://www.apache.org/licenses/LICENSE-2.0
 
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/
#pragma once
#include <atomic>
#include <condition_variable>
#include <list>
#include <mutex>
#include <unordered_map>
#include <thread>
#include "common/utils/spinlock.hpp"

namespace native {
namespace observer {

template <class message_type>
struct actor {
    using message_value_type = message_type;
    using storage_t = std::list<message_value_type>;
    using key_t = size_t;
    using core_t = std::function<void(storage_t& to_do_list)>;

    template <class Function, class... Args>
    actor(key_t actor_id, Function&& f, Args&&... args)
            : function(std::bind(std::forward<Function>(f),
                                 std::forward<Args>(args)...,
                                 std::placeholders::_1)),
              stop(false),
              processing(&actor<message_type>::run, this),
              id(actor_id) {}

    virtual ~actor() {
        stop.store(true);
        if (processing.joinable()) {
            processing.join();
        }
    }

    key_t get_id() const {
        return id;
    }

    template <class typed_message_t>
    void start_job(typed_message_t&& m) {
        {
            std::unique_lock<std::mutex> l(mutex);
            messages.push_back(std::forward<typed_message_t>(m));
            condition.notify_all();
        }
    }

private:
    core_t function;
    storage_t messages;
    std::condition_variable condition;
    std::mutex mutex;

    std::atomic<bool> stop;
    std::thread processing;
    key_t id;

    virtual void run() {
        while (!stop.load()) {
            storage_t to_do_list;
            {
                std::unique_lock<std::mutex> lk(mutex);
                condition.wait(lk, [this]() {
                    return !messages.empty();
                });

                to_do_list.splice(to_do_list.end(), messages);
            }

            function(to_do_list);
        }
    }
};

template <class message_type, class mailbox_message_type>
struct subscribed_actor : public actor<message_type> {
    using base_t = actor<message_type>;
    using self_t = subscribed_actor<message_type, mailbox_message_type>;
    using mailbox_message_t = mailbox_message_type;

    struct mailbox_message_storage_t {
        std::list<mailbox_message_t> container;
        ccl_spinlock lock;
        std::atomic<size_t> messages_count;
    };

    using recipient_storage_t = std::map<key_t, self_t*>;
    using mailbox_table_t = std::unordered_map<key_t, std::unique_ptr<mailbox_message_storage_t>>;

    template <class Function, class... Args>
    subscribed_actor(key_t actor_id, Function&& f, Args&&... args)
            : base_t(actor_id, std::forward<Function>(f), std::forward<Args>(args)..., this) {}

    virtual ~subscribed_actor() {}

    void subscribe_on(subscribed_actor<message_type, mailbox_message_t>* act) {
        if (!act) {
            return;
        }

        // rememeber as recipient
        {
            std::unique_lock<ccl_spinlock> lock(recipients_lock);
            recipients[act->get_id()] = act;
        }
        act->subscribe_on(this);

        // initialize message table
        {
            std::unique_lock<ccl_spinlock> lock(table_lock);
            inner_message_table[act->get_id()].reset(new mailbox_message_storage_t);

            // increase subscriptions count
            subscriptions_table_size.fetch_add(1);
        }
    }

    template <class... message_args>
    void put_message(key_t sender_id, size_t topic_id, message_args&&... args) {
        typename mailbox_table_t::iterator recipient_table_it;
        {
            std::unique_lock<ccl_spinlock> l(table_lock);
            recipient_table_it = inner_message_table.find(sender_id);
            if (recipient_table_it == inner_message_table.end()) {
                throw std::runtime_error("Unregistered recipient");
            }
        }

        // increase total messages count before
        mailbox_message_counter.fetch_add(1);

        std::unique_ptr<mailbox_message_storage_t>& mailbox = recipient_table_it->second;
        (void)topic_id;
        {
            std::unique_lock<ccl_spinlock> l(mailbox->lock);
            mailbox->container.emplace_back(std::forward<message_args>(args)...);

            // increase actual sedner message count
            mailbox->messages_count.fetch_add(1);
        }
    }

    size_t get_subscriptions_count() const {
        return subscriptions_table_size.load();
    }

    size_t get_mailbox_messages_count() const {
        return mailbox_message_counter.load();
    }

    void get_mailbox_messages(key_t sender_id,
                              size_t topic_id,
                              std::list<mailbox_message_t>& messages) {
        typename mailbox_table_t::iterator recipient_table_it;
        {
            std::unique_lock<ccl_spinlock> l(table_lock);
            recipient_table_it = inner_message_table.find(sender_id);
            if (recipient_table_it == inner_message_table.end()) {
                throw std::runtime_error("Unregistered recipient");
            }
        }

        std::unique_ptr<mailbox_message_storage_t>& mailbox = recipient_table_it->second;
        (void)topic_id;
        {
            // check on message existence from sender
            if (mailbox->messages_count.load()) {
                std::unique_lock<ccl_spinlock> l(mailbox->lock);
                mailbox->container.swap(messages);

                // decreae total mesages count
                size_t sender_messages_read_count = mailbox->messages_count.exchange(0);
                mailbox_message_counter.fetch_sub(sender_messages_read_count);
            }
        }
    }

private:
    recipient_storage_t recipients;
    ccl_spinlock recipients_lock;

    mailbox_table_t inner_message_table;
    ccl_spinlock table_lock;
    std::atomic<size_t> subscriptions_table_size;

    std::atomic<size_t> mailbox_message_counter;
};
} // namespace observer
} // namespace native
