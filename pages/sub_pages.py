def get_tabbed_pages():
    # define sub pages
    subpage1 = gr.Interface(
        process_req1,
        inputs=[
            gr.TextArea(placeholder="Input text"),
        ],
        outputs=[
            gr.TextArea(placeholder="Output text"),
        ],
        title="Sub Page 1",
    )
    subpage2 = gr.Interface(
        process_req2,
        inputs=[
            gr.TextArea(placeholder="Input text"),
        ],
        outputs=[
            gr.TextArea(placeholder="Output text"),
        ],
        title="Sub Page 2",
    )
    subpage3 = gr.Interface(
        process_req3,
        inputs=[
            gr.TextArea(placeholder="Input text"),
        ],
        outputs=[
            gr.TextArea(placeholder="Output text"),
        ],
        title="Sub Page 3",
    )

    # define tabbed interface
    tabs = gr.Interface(
        None,
        [
            gr.Interface(
                subpage1,
                title="Tab 1"
            ),
            gr.Interface(
                subpage2,
                title="Tab 2"
            ),
            gr.Interface(
                subpage3,
                title="Tab 3"
            ),
        ],
        title="Tabbed Pages",
        layout="vertical",
        theme="compact"
    )
    return tabs