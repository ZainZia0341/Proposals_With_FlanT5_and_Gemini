from google.generativeai import configure, GenerativeModel
from langchain.prompts import PromptTemplate

GOOGLE_API_KEY = "AIzaSyDHnFpEbZzeDSoFWBqc1CReL8DX8NzaHFk"
configure(api_key=GOOGLE_API_KEY)
model = GenerativeModel('gemini-1.5-pro')

# Function to split the input into manageable chunks
def split_text(text, max_length=2048):
    words = text.split()
    segments = []
    current_segment = []
    current_length = 0

    for word in words:
        word_length = len(word) + 1  # +1 for the space
        if current_length + word_length > max_length:
            segments.append(' '.join(current_segment))
            current_segment = []
            current_length = 0
        current_segment.append(word)
        current_length += word_length

    if current_segment:
        segments.append(' '.join(current_segment))

    return segments

# Function to process each segment and combine the results
def llm_and_prompt(prompt_text):
    prompt_template = PromptTemplate(
    template="""You are an expert proposal writer I will provide you job description and detail about how to write proposal.
        These followings are the rules/principles you need to follow when writing a proposal about a job.
        Write in conversational tone *
        Description: Write the proposal in a friendly and engaging manner so the client feels connected and interested.
        Example: Hey Joe, I totally get what you are talking about. Your idea is a bit challenging, but it is definitely doable. I recently worked on a similar project and can bring that experience to the table.


        Write in 1st person form *
        Description: Use first-person pronouns like 'I' and 'my' to make the proposal more personal and direct.
        Example: I have over 10 years of experience in software development, and I specialize in the MERN stack.


        Write in creative way *
        Description: Make the proposal stand out by being original and engaging, using creative language and presentation.
        Example: Imagine transforming your ideas into a seamless, high-performing application that not only meets but exceeds your expectations. That's what I bring to the table.


        Find pain point and address them *
        Description: Identify the client's specific challenges and offer targeted solutions to address them.
        Example: I understand that poor communication and slow, buggy code deployments have been a major frustration for you. I propose a dedicated project manager and robust CI/CD pipelines to streamline this process.


        Upwork guidelines *
        Description: Follow Upworks best practices for proposals, including clarity, professionalism, and adherence to project requirement, must follow its term and condition and restrictions
        Example: do not include email or do not talk about any freelancing platform


        End words, like regards, name *
        Description: Close the proposal with a polite and professional sign-off.
        Example: Looking forward to collaborating, Best regards,Rafae


        Recent relevant projects *
        Description: Highlight projects that are similar to the clients needs to show relevant experience.
        Example: Recently, I developed an AI-driven application that generates jokes from news headlines using Streamlit and HuggingFace FLAN-T5 model.


        Relevant skills *
        Description: List skills that are directly applicable to the project.
        Example: My skills include MERN stack development, AWS and Amplify, Twilio, Apify API, and Stripe integration.


        Years of experience *
        Description: Mention how long you have been working in your field to build credibility.
        Example: With over 10 years of experience in software development, I bring a wealth of knowledge to your project.


        Client's Name or Business (if name available then *)
        Description: Personalize the proposal by using the clients name or business name.
        Example: Hi [Client's Name], I am excited about the opportunity to work with [Client's Business].


        Understanding of Client's Needs
        Description: Show that you understand what the client is looking for and why it is important to them.
        Example: I understand that you need a reliable partner who can ensure seamless communication and efficient project management to avoid any slow and buggy deployments.


        Unique Selling Proposition (USP)
        Description: Highlight what makes you different and why the client should choose you.
        Example: What sets me apart is my proactive communication and streamlined project management approach, ensuring that your project is always on track.


        Proposed Solution
        Description: Provide a clear and concise plan on how you will address the clients needs.
        Example: I propose setting up a robust CI/CD pipeline and maintaining regular updates through a dedicated project manager to ensure smooth progress and high-quality deployments.


        Add Technical Expertise
        Description: Detail your technical skills and expertise relevant to the project.
        Example: I have extensive experience with the MERN stack, AWS, Amplify, and integrating tools like Twilio, Apify API, and Stripe.


        Add Case Studies
        Description: Include examples of past work that illustrate your ability to handle similar projects.
        Example: In a previous project, I developed a highly interactive AI application that successfully integrated multiple APIs and handled large data sets efficiently.


        Client Testimonials
        Description: Share positive feedback from past clients to build trust and credibility.
        Example: One of my clients said, 'Working with [Your Name] was a game-changer for our project. Their communication was top-notch, and they delivered high-quality work on time.'


        Brief Work Plan to do the project
        Description: Outline the steps you will take to complete the project, providing a clear roadmap.
        Example: 1. Initial consultation and requirement analysis 2. Development of a communication plan 3. Setup of CI/CD pipeline 4. Development and integration phases 5. Testing and quality assurance 6. Deployment and post-delivery support.


        Tools and Technologies will use
        Description: List the specific tools and technologies you will use to execute the project.
        Example: I will use the MERN stack, AWS and Amplify, Twilio, Apify API, Stripe, Streamlit, and Hugging Face Transformers.


        Post Delivery Support
        Description: Describe the support you will offer after the project is completed.
        Example: I offer comprehensive post-delivery support, including bug fixes, performance optimizations, and feature enhancements.


        Link to Portfolio *
        Description: Provide a link to your portfolio to showcase your previous work.
        Example: You can view my portfolio [here](#).


        Github link *
        Description: Include a link to a relevant GitHub project to demonstrate your coding skills.
        Example: Check out a relevant project on my GitHub [here](https://github.com/Rafaekhatri11/AI_News_Headline_Jokes).


        Problem-solving approach
        Description: Explain your method for addressing and solving the clients problems.
        Example: My approach focuses on identifying the root causes of issues and developing solutions that enhance user experience and system performance.


        Commitment to Client's Success
        Description: Emphasize your dedication to helping the client achieve their goals.
        Example: Your success is my priority. I'm dedicated to delivering high-quality work that helps you achieve your objectives.


        Commitment to Deadlines
        Description: Assure the client of your ability to meet deadlines consistently.
        Example: I am committed to delivering projects on time without compromising on quality, backed by my robust project management practices.


        Show you understand Client's Business or problem
        Description: Demonstrate that you have a good grasp of the clients industry or specific challenges.
        Example: I understand that in your industry, seamless integration and reliable performance are crucial. My experience with similar projects ensures that I can meet these needs effectively.


        Ask Questions as call to action *
        Description: Engage the client by asking questions that encourage further discussion.
        Example: Are there any specific features or functionalities you want to prioritize? Lets discuss how we can make your project a success.
        

        Take this as an examples job description
                
        We already have a live production site with thousands of profiles, I will give the details when we message.   Basically I have contracted out the work to several large offshore Indian companies, but have so far been very unhappy with the lack of communication, and too many project managers making the code deployment very slow and even at times buggy.   
        This is a huge opportunity for the right agency............

        Our current tech stack is MERN, and we are using AWS and Amplify to deploy the code.   We are also using Twilio for the messaging, and Apify API, and Stripe for payments.   Also integrated with Salesforce, although I have a dev for that portion already.


        And this is the proposal for this example job description.

        I'm excited about the opportunity to work with you on your MERN stack project. I understand how crucial effective communication and efficient project management are for the success of your project. With over 10 years in software development, I've honed my skills in managing complex tech stacks, including MERN, AWS, Amplify, Twilio, Apify API, Stripe, and Salesforce. One of my recent projects was an AI-driven application that generates jokes from news headlines using Streamlit and Hugging Face's FLAN-T5 model. You can check out this project here.

        I believe in proactive communication, keeping you updated regularly through Slack, Zoom, or email. To streamline our interactions, I'll be your single point of contact, ensuring your needs are met efficiently without the hassle of dealing with multiple project managers. I also implement robust CI/CD pipelines for efficient code deployment and rigorous testing protocols to maintain high-quality standards.

        I'm committed to delivering your project on time without compromising on quality. My approach includes a detailed work plan and clear project milestones, ensuring every phase is executed flawlessly. Post-delivery, I offer comprehensive support to ensure your application runs smoothly, including bug fixes and performance optimizations.

        My problem-solving approach focuses on understanding and addressing root causes, enhancing user experience and system performance. I'm dedicated to your success and look forward to turning your challenges into success stories. I'd love to discuss any specific features or pain points you have in mind and work together to achieve outstanding results.

        Best regards,
        Rafae.

        This is the job description make a proposal like example
         
        {prompt_text}""",
    input_variables=["prompt_text"]
    )

    segments = split_text(prompt_text)
    combined_response = ""
    
    for segment in segments:
        formatted_prompt = prompt_template.format(prompt_text=segment)
        response = model.generate_content(formatted_prompt)
        combined_response += response.text + "\n"

    print(combined_response)
    return combined_response

# Example usage:
job_description = """
We already have a live production site with thousands of profiles, I will give the details when we message. Basically I have contracted out the work to several large offshore Indian companies, but have so far been very unhappy with the lack of communication, and too many project managers making the code deployment very slow and even at times buggy. This is a huge opportunity for the right agency............

Our current tech stack is MERN, and we are using AWS and Amplify to deploy the code. We are also using Twilio for the messaging, and Apify API, and Stripe for payments. Also integrated with Salesforce, although I have a dev for that portion already.
"""
llm_and_prompt(job_description)
