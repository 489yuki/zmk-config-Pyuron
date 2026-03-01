#define DT_DRV_COMPAT zmk_input_processor_trackball_to_key

#include <zephyr/device.h>
#include <zephyr/kernel.h>
#include <zephyr/sys/util.h>
#include <zephyr/logging/log.h>
#include <stdlib.h>

#include <drivers/input_processor.h>
#include <zmk/events/keycode_state_changed.h>

LOG_MODULE_DECLARE(zmk, CONFIG_ZMK_LOG_LEVEL);

#define HID_USAGE_KEY_PAGE 0x07

struct trackball_to_key_config {
    int threshold;
    int deadzone;
    int max_per_event;
    uint32_t right_keycode;
    uint32_t left_keycode;
};

struct trackball_to_key_data {
    int right_accum;
    int left_accum;
};

static inline int tap_key(uint32_t keycode) {
    int ret;

    ret = raise_zmk_keycode_state_changed((struct zmk_keycode_state_changed){
        .usage_page = HID_USAGE_KEY_PAGE,
        .keycode = keycode,
        .implicit_modifiers = 0,
        .explicit_modifiers = 0,
        .state = true,
        .timestamp = k_uptime_get(),
    });
    if (ret < 0) {
        LOG_ERR("Failed to press key 0x%02x: %d", keycode, ret);
        return ret;
    }

    ret = raise_zmk_keycode_state_changed((struct zmk_keycode_state_changed){
        .usage_page = HID_USAGE_KEY_PAGE,
        .keycode = keycode,
        .implicit_modifiers = 0,
        .explicit_modifiers = 0,
        .state = false,
        .timestamp = k_uptime_get(),
    });
    if (ret < 0) {
        LOG_ERR("Failed to release key 0x%02x: %d", keycode, ret);
    }

    return ret;
}

static int trackball_to_key_handle_event(const struct device *dev,
                                         struct input_event *event,
                                         uint32_t param1, uint32_t param2,
                                         struct zmk_input_processor_state *state) {
    const struct trackball_to_key_config *cfg = dev->config;
    struct trackball_to_key_data *data = dev->data;

    if (event->type != INPUT_EV_REL) {
        return 0;
    }

    /* Consume Y axis on this layer (no pointer movement) */
    if (event->code == INPUT_REL_Y) {
        event->value = 0;
        return 0;
    }

    /* Only process X axis */
    if (event->code != INPUT_REL_X) {
        return 0;
    }

    int dx = event->value;

    /* Deadzone: ignore tiny movements */
    if (abs(dx) <= cfg->deadzone) {
        event->value = 0;
        return 0;
    }

    if (dx > 0) {
        /* Right movement -> Tab */
        data->left_accum = 0;
        data->right_accum += dx;

        int fires = 0;
        while (data->right_accum >= cfg->threshold && fires < cfg->max_per_event) {
            tap_key(cfg->right_keycode);
            data->right_accum -= cfg->threshold;
            fires++;
        }
    } else {
        /* Left movement -> Shift */
        data->right_accum = 0;
        data->left_accum += (-dx);

        int fires = 0;
        while (data->left_accum >= cfg->threshold && fires < cfg->max_per_event) {
            tap_key(cfg->left_keycode);
            data->left_accum -= cfg->threshold;
            fires++;
        }
    }

    /* Consume the REL_X event */
    event->value = 0;
    return 0;
}

static struct zmk_input_processor_driver_api trackball_to_key_driver_api = {
    .handle_event = trackball_to_key_handle_event,
};

static int trackball_to_key_init(const struct device *dev) {
    return 0;
}

#define TRACKBALL_TO_KEY_INST(n)                                               \
    static struct trackball_to_key_data data_##n = {};                         \
    static const struct trackball_to_key_config config_##n = {                 \
        .threshold = DT_INST_PROP(n, threshold),                               \
        .deadzone = DT_INST_PROP(n, deadzone),                                 \
        .max_per_event = DT_INST_PROP(n, max_per_event),                       \
        .right_keycode = DT_INST_PROP(n, right_keycode),                       \
        .left_keycode = DT_INST_PROP(n, left_keycode),                         \
    };                                                                         \
    DEVICE_DT_INST_DEFINE(n, trackball_to_key_init, NULL, &data_##n,           \
                          &config_##n, POST_KERNEL,                            \
                          CONFIG_KERNEL_INIT_PRIORITY_DEFAULT,                 \
                          &trackball_to_key_driver_api);

DT_INST_FOREACH_STATUS_OKAY(TRACKBALL_TO_KEY_INST)
